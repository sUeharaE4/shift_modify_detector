from logging import INFO, DEBUG
import argparse
import sys
import os
from os.path import isfile
import numpy as np
import cv2
from tqdm import tqdm
import csv

import util
from log_mod import modify_logger
# ログの追加フォーマット
extra_args = {}
extra_args['tab'] = '\t'

detect_logger_cls = modify_logger.ModifyLogger()
logger = detect_logger_cls.create_logger(__name__, INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='テンプレート画像と入力画像の差分抽出するスクリプト.' +
                                     '2画像の差分を取ることで記入箇所を抽出する.2画像が完全に重なることはないため、' +
                                     '若干膨張(dilation)させてから差分を取るので閾値調整が必要.')
    # 入力画像と出力先
    parser.add_argument('--template_path', default=None, type=str,
                        help='テンプレート画像のパス.')
    parser.add_argument('--pair_path', default=None, type=str,
                        help='差分抽出する画像のパス.')
    parser.add_argument('--output_dir', default='detect_results/', type=str,
                        help='差分情報を記載したファイルの出力先.')
    # ズレ修正でディレクトリを指定するか
    # TODO 将来的にはサブコマンドで実装してほしい
    parser.add_argument('--detect_multi', action='store_true',
                        help='差分抽出対象を1枚の画像ではなくディレクトリにする場合に使用する.')
    parser.add_argument('--detect_dir', default='modify/', type=str,
                        help='差分抽出する画像が格納されたディレクトリのパス.--template_path以外すべて抽出対称にする.')
    # 比較画像を作成するか
    parser.add_argument('--create_diff', action='store_true',
                        help='抽出した差分を枠で囲った画像を出力する場合は使用する.')
    # 閾値
    parser.add_argument('--threthold_BW', default=200, type=int,
                        help='白と見なす画素値.1～255で通常は200以上。灰色の領域があれば150等調整してください.')
    parser.add_argument('--drop_min_length', default=30, type=int,
                        help='抽出した矩形領域のうち、小さすぎるため除去する辺の長さ(ピクセル).')
    parser.add_argument('--mask_kernel_size', default=5, type=int,
                        help='差分を取る前にdilation, erosionするkernelのサイズ.1,3,5程度.')
    parser.add_argument('--mask_dilation_itr', default=4, type=int,
                        help='枠線等を除去するためのdilation回数.多すぎると記入部分も失う')
    parser.add_argument('--text_kernel_size', default=3, type=int,
                        help='記入箇所を膨張させるkernelのサイズ.1,3程度.')
    parser.add_argument('--text_dilation_itr', default=20, type=int,
                        help='記入箇所を膨張させ、外接矩形を抽出するためのdilation回数')
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
    if args.pair_path is None and not args.detect_multi:
        print('比較画像のパスを指定してください。またはディレクトリを指定してください.')
        print('例1：--pair_path input/other.jpg')
        print('例2：--detect_multi --detect_dir modify/')
        return False
    if args.detect_multi and not os.path.exists(args.detect_dir):
        print('detect_dirが存在しません')
        return False
    # 問題なければTrueを返却する
    return True


if __name__ == '__main__':
    # 入力チェック(型までは見ない)
    args = parse_args()
    valid_input = check_args(args)
    if not valid_input:
        sys.exit(1)

    if args.debug:
        logger = detect_logger_cls.setLevelUtil(logger, DEBUG)

    BASE_IMG             = args.template_path
    OUTPUT_DIR           = args.output_dir
    DROP_MIN_LENGTH      = args.drop_min_length
    MASK_KERNEL_SIZE     = args.mask_kernel_size
    MASK_DILATION_KERNEL = np.ones((MASK_KERNEL_SIZE, MASK_KERNEL_SIZE), np.uint8)
    MASK_DILATION_ITER   = args.mask_dilation_itr
    TEXT_KERNEL_SIZE     = args.text_kernel_size
    TEXT_DILATION_KERNEL = np.ones((TEXT_KERNEL_SIZE, TEXT_KERNEL_SIZE), np.uint8)
    TEXT_DILATION_ITER   = args.text_dilation_itr

    DIR_SEP = os.sep

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    if args.detect_multi:
        pair_img_list = [path for path in os.listdir(args.detect_dir) if not os.path.isdir(path)]
        # TODO 指定ディレクトリ直下のファイルだけをlistに格納するように変更する
        pair_img_list = list(set(pair_img_list) - set([BASE_IMG.split(DIR_SEP)[-1], 'diff']))
        pair_img_list = [os.path.join(args.detect_dir, img) for img in pair_img_list]
    else:
        pair_img_list = [args.pair_path]
    logger.debug('pair_img_list : ' + str(pair_img_list), extra=extra_args)

    for pair_img_path in tqdm(pair_img_list):
        logger.debug('target_img : ' + pair_img_path, extra=extra_args)
        base_img, pair_img = util.read_base_pair_imgs(BASE_IMG, pair_img_path, args.threthold_BW)
        # 画素の差分をでマスク画像を生成
        mask_dilation = util.calc_mask(base_img, pair_img, threshold=args.threthold_BW,
                                       with_dilation=True, kernel=MASK_DILATION_KERNEL, itr=MASK_DILATION_ITER)
        tmp = cv2.morphologyEx(mask_dilation, cv2.MORPH_OPEN, (1, 1))
        dilation = cv2.dilate(tmp, MASK_DILATION_KERNEL, iterations=3)
        # 画素の差分を抽出、差分の膨張
        img_masked = cv2.bitwise_and(pair_img, dilation)
        img_masked_dilation = cv2.dilate(img_masked, TEXT_DILATION_KERNEL, iterations=TEXT_DILATION_ITER)
        # 0,1 から0,255へ
        img_masked_dilation = img_masked_dilation * 255
        # 矩形領域の抽出
        contours, hierarchy = cv2.findContours(img_masked_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 1辺が小さすぎる矩形の除外
        min_area = DROP_MIN_LENGTH
        large_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > DROP_MIN_LENGTH and h > DROP_MIN_LENGTH:
                large_contours.append(cnt)
        # csvの書き込み
        output_file_name = pair_img_path.split(DIR_SEP)[-1].split('.')[0]
        output_file_path = os.path.join(OUTPUT_DIR, output_file_name + '.csv')
        logger.debug('output_file_path : ' + output_file_path, extra=extra_args)
        with open(output_file_path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['x', 'y', 'width', 'height'])
            for large_cnt in large_contours:
                x, y, width, height = cv2.boundingRect(large_cnt)
                writer.writerow([x, y, width, height])

        # 画像へ矩形の書き込み
        if args.create_diff:
            pair_img = cv2.imread(pair_img_path)
            img_type = '.' + pair_img_path.split(DIR_SEP)[-1].split('.')[-1]
            output_img_path = os.path.join(OUTPUT_DIR, output_file_name + img_type)
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                written_img_rectangle = cv2.rectangle(pair_img,
                                                      (x, y), (x + w, y + h),
                                                      (0, 255, 0), 5)
            cv2.imwrite(output_img_path, pair_img)
        logger.debug('your inputs : ' + str(args), extra=extra_args)
