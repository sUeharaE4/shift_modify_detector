import argparse
import sys
import os
from os.path import isfile
import numpy as np
import cv2
from tqdm import tqdm

import util
import ripoc
# from shift_modification.log_mod import modify_logger
sys.path.append('./log_mod/')
from log_mod import modify_logger
from logging import INFO, DEBUG
# ログの追加フォーマット
extra_args = {}
extra_args['tab'] = '\t'

modify_logger_cls = modify_logger.ModifyLogger()
logger = modify_logger_cls.create_logger(__name__, INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='テンプレート画像と入力画像のズレを修正するスクリプト.')
    # 入力画像と出力先
    parser.add_argument('--template_path', default=None, type=str,
                        help='テンプレート画像のパス.ズレ修正するための基準画像.')
    parser.add_argument('--pair_path', default=None, type=str,
                        help='ズレを修正する画像のパス.')
    parser.add_argument('--output_dir', default='output/', type=str,
                        help='ズレ修正した画像の出力先.')
    # ズレ修正でディレクトリを指定するか
    # TODO 将来的にはサブコマンドで実装してほしい
    parser.add_argument('--modify_multi', action='store_true',
                        help='ズレ修正対象を1枚の画像ではなくディレクトリにする場合に使用する.')
    parser.add_argument('--modify_dir', default='input/', type=str,
                        help='ズレ修正する画像が格納されたディレクトリのパス.--template_path以外すべて修正対称にする')
    # 比較画像を作成するか
    parser.add_argument('--create_diff', action='store_true',
                        help='ズレ修正前後とテンプレート画像を並べた比較画像を生成する場合に使用する.')
    # 閾値
    parser.add_argument('--threthold_BW', default=200, type=int,
                        help='白と見なす画素値.1～255で通常は200以上。灰色の領域があれば150等調整してください.')
    parser.add_argument('--mag_scale', default=100, type=int,
                        help='RIPOCする際のmagnitude_scale. よくわからなければ指定不要です.')
    # Debug log を出力するか
    parser.add_argument('--debug', action='store_true',
                        help='debug log をコンソールに出力するか.出力する場合進捗表示が崩れる.')

    args = parser.parse_args()

    return args


def check_args(args):
    """
    実行時引数の整合性チェック.
    Parameters
    ----------
    args :
        parse_args()した実行時引数

    Returns
    -------
    check_result : bool
        整合性エラーが生じた時点でFalseが返却される.
    """
    if args.template_path is None:
        print('テンプレート画像のパスを指定してください.例：--template_path input/template.jpg')
        return False
    if not isfile(args.template_path):
        print('テンプレート画像が存在しません.パスを確認してください.')
        return False
    if args.pair_path is None and not args.modify_multi:
        print('比較画像のパスを指定してください。またはディレクトリを指定してください.')
        print('例1：--pair_path input/other.jpg')
        print('例2：--modify_multi --modify_dir input/')
        return False
    if args.modify_multi and not os.path.exists(args.modify_dir):
        print('modify_dirが存在しません')
        return False
    # 問題なければTrueを返却する
    return True


def __binarize(img, threshold):
    """
    画像を2値化する.アドレス参照して変換するので注意.
    Parameters
    ----------
    img : numpy.ndarray
        入力画像
    threshold : int
        2値化する閾値. 0～255

    Returns
    -------

    """
    img[img < threshold] = 0
    img[img >= threshold] = 255


def read_base_pair_imgs(base_img_path, pair_img_path, threshold):
    base_img = util.exchange_black_white(cv2.imread(base_img_path, 0))
    pair_img = util.exchange_black_white(cv2.imread(pair_img_path, 0))
    # baseとpairのサイズを合わせる
    pair_img = util.expand_cut2base_size(base_img, pair_img)
    # 2値化と形式変換
    for img in [base_img, pair_img]:
        __binarize(img, threshold)
    return [base_img, pair_img]


def expand_imgs(base_img, pair_img):
    base_img = util.expand2square(base_img, [0, 0])
    pair_img = util.expand2square(pair_img, [0, 0])

    height, width = base_img.shape[0:2]
    base_img = np.asarray(base_img, dtype=np.float64)
    base_img = base_img[slice(height), slice(width)]
    pair_img = np.asarray(pair_img, dtype=np.float64)
    pair_img = pair_img[slice(height), slice(width)]

    return [base_img, pair_img]


def resize_imgs(base_img, pair_img, resize_shape):
    base_img_resize = cv2.resize(base_img, resize_shape)
    pair_img_resize = cv2.resize(pair_img, resize_shape)
    return [base_img_resize, pair_img_resize]


def rotate_modify(base_img, pair_img):
    """
    回転方向のズレを修正する.
    Parameters
    ----------
    base_img : numpy.ndarray
        テンプレート画像
    pair_img : numpy.ndarray
        修正対象画像

    Returns
    -------
    modified_img : numpy.ndarray
        回転方向のズレを修正した画像.
    """
    # 512,512にリサイズ(計算効率・閾値の調整しやすさでこのサイズにした)
    resize_base_img, resize_pair_img = resize_imgs(base_img, pair_img, RESIZE_SHAPE)
    row, col = resize_base_img.shape[0:2]
    hrow = int(row/2)
    center = tuple(np.array(resize_base_img.shape) / 2)
    logger.debug('resize_size : ' + str(resize_base_img.shape[0:2]), extra=extra_args)

    # 対数極座標変換と回転・拡大の推定
    FLP, GLP = ripoc.logpolar_module(resize_base_img, resize_pair_img, MAG_SCALE)

    row_shift, col_shift, peak_map = ripoc.fft_coreg_LP(FLP, GLP)
    angle_est = - row_shift / (hrow) * 180
    scale_est = 1.0 - col_shift / MAG_SCALE

    # rotate slave
    rotMat = cv2.getRotationMatrix2D(center, angle_est, 1.0)
    g_coreg = cv2.warpAffine(resize_pair_img, rotMat, resize_pair_img.shape, flags=cv2.INTER_LANCZOS4)
    # scale slave
    g_coreg_tmp = cv2.resize(g_coreg, None, fx=scale_est, fy=scale_est, interpolation=cv2.INTER_LANCZOS4)
    row_coreg_tmp = g_coreg_tmp.shape[0]
    col_coreg_tmp = g_coreg_tmp.shape[1]
    g_coreg = np.zeros((row, col))
    if row_coreg_tmp == row:
        g_coreg = g_coreg_tmp
    elif row_coreg_tmp > row:
        g_coreg = g_coreg_tmp[slice(row), slice(col)]
    else:
        g_coreg[slice(row_coreg_tmp), slice(col_coreg_tmp)] = g_coreg_tmp

    # estimate translation & translate slave
    row_shift, col_shift, peak_map, g_coreg = ripoc.fft_coreg_trans(resize_base_img, g_coreg)
    # check estimates
    logger.debug('RIPOC Results', extra=extra_args)
    logger.debug('rotate angle : ' + str(angle_est), extra=extra_args)
    logger.debug('scale : '        + str(scale_est), extra=extra_args)

    tmp_height, tmp_width = expand_pair_img.shape
    center = (tmp_width / 2, tmp_height / 2)
    trans = cv2.getRotationMatrix2D(center, angle_est, 1.0)
    modified_img = cv2.warpAffine(pair_img, trans, (tmp_width, tmp_height))

    return modified_img


def shift_modify(base_img, pair_img):
    """
    水平垂直方向のズレを修正する.
    Parameters
    ----------
    base_img : numpy.ndarray
        テンプレート画像
    pair_img : numpy.ndarray
        テンプレート画像

    Returns
    -------
    modified_img : numpy.ndarray
        水平垂直方向のズレを修正した画像.
    """
    height, width = pair_img.shape
    shift, etc = cv2.phaseCorrelate(base_img, pair_img)
    dx, dy = shift
    logger.debug('POC Results', extra=extra_args)
    logger.debug('x_shift : ' + str(dx), extra=extra_args)
    logger.debug('y_shift : ' + str(dy), extra=extra_args)

    trans = np.float32([[1, 0, -1 * dx], [0, 1, -1 * dy]])
    modified_img = cv2.warpAffine(rotate_expand_pair_img, trans, (width, height))
    modified_img = util.exchange_black_white(modified_img)[0:default_height, 0:default_width]

    return modified_img


def write_ruled_line(img, interval=100):
    """
    画像に罫線を追加する.
    Parameters
    ----------
    img : numpy.ndarray
        画像
    interval : int
        罫線の間隔

    Returns
    -------
    ruled_img : numpy.ndarray
        画像に罫線を追加した画像.
    """
    ruled_img = img.copy()
    for i in range(ruled_img.shape[0] // interval - 1):
        line_pos = interval * (i + 1)
        ruled_img[line_pos - 2:line_pos + 2, :] = (0, 255, 0)
    for i in range(ruled_img.shape[1] // interval - 1):
        line_pos = interval * (i + 1)
        ruled_img[:, line_pos - 2:line_pos + 2] = (0, 255, 0)
    return ruled_img


if __name__ == '__main__':
    # 入力チェック(型までは見ない)
    args = parse_args()
    valid_input = check_args(args)
    if not valid_input:
        sys.exit(1)

    if args.debug:
        logger = modify_logger_cls.setLevelUtil(logger, DEBUG)

    BASE_IMG   = args.template_path
    OUTPUT_DIR = args.output_dir
    RESIZE_LEN = 512
    MAG_SCALE  = args.mag_scale
    RESIZE_SHAPE = (RESIZE_LEN, RESIZE_LEN)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if args.modify_multi:
        pair_img_list = os.listdir(args.modify_dir)
        pair_img_list = list(set(pair_img_list) - set([BASE_IMG.split('/')[-1]]))
        pair_img_list = [os.path.join(args.modify_dir, img) for img in pair_img_list]
    else:
        pair_img_list = [args.pair_path]
    logger.debug('pair_img_list : ' + str(pair_img_list), extra=extra_args)

    for pair_img_path in tqdm(pair_img_list):
        logger.debug('target_img : ' + pair_img_path, extra=extra_args)
        base_img, pair_img = read_base_pair_imgs(BASE_IMG, pair_img_path, args.threthold_BW)
        default_height, default_width = base_img.shape[0:2]
        logger.debug('default_size : ' + str(base_img.shape[0:2]), extra=extra_args)

        expand_base_img, expand_pair_img = expand_imgs(base_img, pair_img)
        height, width = expand_base_img.shape
        logger.debug('expand_size : ' + str(expand_base_img.shape[0:2]), extra=extra_args)

        # 回転方向の修正
        rotate_expand_pair_img = rotate_modify(expand_base_img, expand_pair_img)
        # 回転修正したのでPOC
        modified_img = shift_modify(expand_base_img, rotate_expand_pair_img)
        logger.debug('modified_size : ' + str(modified_img.shape[0:2]), extra=extra_args)

        output_img_name, output_img_ext = pair_img_path.split('/')[-1].split('.')
        output_img_ext = '.' + output_img_ext
        output_img_path = os.path.join(OUTPUT_DIR, output_img_name + '_modify' + output_img_ext)
        logger.debug('output_path : ' + output_img_path, extra=extra_args)
        cv2.imwrite(output_img_path, modified_img)

        if args.create_diff:
            # 一度形式を変更してしまった画像をもとに戻す
            base_img, pair_img = read_base_pair_imgs(BASE_IMG, pair_img_path, args.threthold_BW)
            base_img, pair_img = util.exchange_black_white(base_img), util.exchange_black_white(pair_img)
            modified_img = np.asarray(modified_img, dtype=np.uint8)
            logger.debug('hconcat 3 imgs : ' +
                         str(base_img.shape[0:2]) + ' ' + str(base_img.shape[0:2]) + ' ' + str(modified_img.shape[0:2]),
                         extra=extra_args)
            show_img = cv2.hconcat([base_img, pair_img, modified_img])
            # カラー読み込み用
            cv2.imwrite('tmp' + output_img_ext, show_img)
            show_img = cv2.imread('tmp' + output_img_ext)
            os.remove('tmp' + output_img_ext)
            # 罫線追加
            show_img = write_ruled_line(show_img)
            scale = width / show_img.shape[1]
            show_img = cv2.resize(show_img, dsize=None, fx=scale, fy=scale)
            cv2.imwrite(os.path.join(OUTPUT_DIR, output_img_name + '_diff' + output_img_ext), show_img)
