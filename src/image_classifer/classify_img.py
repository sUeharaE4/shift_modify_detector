from logging import INFO, DEBUG
import argparse
from distutils.util import strtobool
import sys
import os
from os.path import isfile
import cv2
from tqdm import tqdm
import pickle
import traceback
import pandas as pd
import glob

import util
from log_mod import modify_logger
# ログの追加フォーマット
extra_args = {}
extra_args['tab'] = '\t'

my_logger_cls = modify_logger.ModifyLogger()
logger = my_logger_cls.create_logger(__name__, INFO)
detector = cv2.AKAZE_create()
DIR_SEP = os.sep


def parse_args():
    parser = argparse.ArgumentParser(
        description='入力画像が与えられたディレクトリ配下の\
                     どのディレクトリに属するか推定する.')
    # 入力と出力先
    parser.add_argument('--img_path', type=str,
                        help='画像のパス.')
    parser.add_argument('--registered_dir', type=str,
                        help='画像の種類ごとにディレクトリ分けした\
                              ルートディレクトリ.')
    parser.add_argument('--score_path', type=str,
                        help='前回計算した比較結果を読み込む場合はパスを指定.'
                             '同じ画像の組み合わせは計算しなくなる.')
    parser.add_argument('--output_path', type=str,
                        help='推定結果ファイルの出力先.')
    # 画像を1枚ではなく複数推定するか
    parser.add_argument('--classify_multi', type=strtobool,
                        help='分類したい画像が複数ある場合True.')
    parser.add_argument('--classify_dir', type=str,
                        help='分類したい画像が格納されたディレクトリのパス.')
    # TODO 明らかにこれは違うとわかっているディレクトリは比較対象外にするオプションが欲しい

    # 比較結果を保存するか
    parser.add_argument('--save_score', type=strtobool,
                        help='画像の類似度を比較した結果を保存する場合は指定.'
                             '次回は同じ組み合わせで計算しない場合利用する.')
    parser.add_argument('--save_path', type=str,
                        help='画像の類似度を比較した結果の保存先.')
    # 2値化
    parser.add_argument('--change_bright', type=strtobool,
                        help='色の影響を受けやすいので閾値を設けて\
                              明るさを操作するか.')
    parser.add_argument('--threthold_W', type=int,
                        help='白と見なす画素値.1～255で通常は200以上。\
                              灰色の領域があれば150等調整してください.')
    parser.add_argument('--threthold_B', type=int,
                        help='黒と見なす画素値.白黒2値化するなら\
                              threthold_Wと同じ値.')
    # 平均を計算して分類する
    parser.add_argument('--calc_mean', type=strtobool,
                        help='画像1枚1枚のスコアで分類するのではなく、\
                              同じ種類の平均で分類する.')

    parser.add_argument('--print_rank', type=int,
                        help='上位何件まで表示するか.')
    # Debug log を出力するか
    parser.add_argument('--debug', type=strtobool,
                        help='debug log をコンソールに出力するか.\
                              出力する場合進捗表示が崩れる.')
    parser.add_argument('--conf_path', type=str,
                        default='../conf/classify.yml',
                        help='設定ファイルのパス.デフォルトは\
                              ../conf/classify.yml.')
    args = parser.parse_args()

    return args


def check_config(config):
    """
    設定値の整合性チェック.

    Parameters
    ----------
    config :
        設定ファイルと実行時引数を読み込んだdict

    Returns
    -------
    check_result : bool
        整合性エラーが生じた時点でFalseが返却される.
    """
    img_path = config['input']['img_path']
    registered_dir = config['input']['registered_dir']
    score_path = config['input']['score_path']
    classify_dir = config['input']['classify_dir']
    classify_multi = config['mode']['classify_multi']
    change_bright = config['options']['change_bright']
    threthold_W = config['options']['threthold_W']
    threthold_B = config['options']['threthold_B']
    if img_path is None and not classify_multi:
        print('画像のパスを指定してください.'
              '例：--img_path unknown/questionnaire_001.jpg')
        return False
    if score_path is not None and not isfile(score_path):
        print('指定した計算結果がありません.パスを確認してください：' +
               score_path)
        return False
    if not os.path.isdir(registered_dir):
        print('事前に画像をディレクトリ分けしたディレクトリがありません.'
              'パスを確認してください :' + registered_dir)
        return False
    if classify_multi and not os.path.exists(classify_dir):
        print('classify_dirが存在しません :' + classify_dir)
        return False
    if change_bright:
        if threthold_W is None:
            print('白と見なす明るさを指定してください. 例：--threthold_W 200')
            return False
        if threthold_B is None:
            print('黒と見なす明るさを指定してください. 例：--threthold_B 200')
            return False
    # 問題なければTrueを返却する
    return True


def chenge_bright(img, threthold_W, threthold_B):
    """
    閾値で画像の明るさを変更する.アドレス参照して変換するのでreturn無し.

    Parameters
    ----------
    img : numpy.ndarray
        画像
    threthold_W : int
        白と見なす明るさ.これ以上は255になる.
    threthold_W : int
        黒と見なす明るさ.これ以上は0になる.

    Returns
    -------

    """
    img[img >= threthold_W] = 255
    img[img <  threthold_B] = 0


def read_pre_calc_score(score_path):
    """
    既に計算済みのスコアを読み込む.

    Parameters
    ----------
    score_path : str
        計算済みスコアのパス.

    Returns
    -------
    score_dict : dict
        計算済みスコアを格納したdict
    """
    with open(score_path, 'rb') as score_f:
        score_dict = pickle.load(score_f)
    return score_dict


def dump_score(score_path, score_dict):
    """
    計算済みのスコアを保存する.

    Parameters
    ----------
    score_path : str
        計算済みスコアを保存するパス.
    score_dict : dict
        計算済みスコア.

    Returns
    -------

    """
    with open(score_path, 'wb') as score_f:
        pickle.dump(score_dict, score_f)


def __get_kp_and_des(img):
    """
    画像の特徴点を抽出する.

    Parameters
    ----------
    img : str
        特徴点抽出したい画像のパス.

    Returns
    -------
    (kp, des) : tuple
        画像の類似度.
    """
    global detector
    (kp, des) = detector.detectAndCompute(img, None)
    return (kp, des)


def get_match_points(unknown_path, known_path, match_dict,
                     threthold_W=None, threthold_B=None):
    """
    与えられた画像ペアの類似度を返す.既に計算済みであればそのまま返却し、
    計算済みでなければ特徴点マッチングで計算する.

    Parameters
    ----------
    unknown_path : str
        分類したい画像のパス.
    known_path : str
        分類が分かっている画像のパス.
    match_dict : dict
        画像の組み合わせと類似度を保存するdict.

    Returns
    -------
    ret : float
        類似度.
    """
    # 既に計算済みであれば計算済みの値を返す
    global DIR_SEP
    if unknown_path in match_dict:
        if known_path in match_dict[unknown_path]:
            return match_dict[unknown_path][known_path]
    # 計算済みでなければ仕方ない
    i1 = cv2.imread(unknown_path, cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(known_path, cv2.IMREAD_GRAYSCALE)
    if threthold_W is not None and threthold_B is not None:
        chenge_bright(i1, threthold_W, threthold_B)
        chenge_bright(i2, threthold_W, threthold_B)

    # サイズが違うなら比較しない
    if i1.shape != i2.shape:
        return 10000000

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    (target_kp, target_des) = __get_kp_and_des(i1)
    (comparing_kp, comparing_des) = __get_kp_and_des(i2)
    try:
        matches = bf.match(target_des, comparing_des)
        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
    except cv2.error:
        print(traceback.format_exc())
        return 10000000

    return ret


def calc_mean(score_dict):
    df = pd.DataFrame.from_dict(score_dict)
    groups = [util.concat_path(item.split(DIR_SEP)[0:-1], DIR_SEP) \
              for item in df.index]
    df['group'] = groups
    mean_dict = {}
    for image in df.drop('group', axis=1).columns:
        logger.debug('image : ' + str(image), extra=extra_args)
        mean_dict[image] = df[~df[image].isnull()]\
                           .groupby('group').mean()[image].to_dict()

    return mean_dict


def write_result(score_dict, output_path, use_mean=False, print_ranks=3):
    """
    比較結果を出力する.

    Parameters
    ----------
    score_dict : dict
        計算済みスコア.
    output_path : str
        出力先.
    use_mean : bool
        比較する際に画像1枚1枚のスコアを見るか、ディレクトリ毎に平均を見るか.
    Returns
    -------

    """
    if use_mean:
        score_dict = calc_mean(score_dict)
    header = ['top_' + str(rank + 1) for rank in range(print_ranks)]
    header.insert(0, 'img')
    result_df = pd.DataFrame(columns=header)
    for img_path in score_dict:
        diff_scores = score_dict[img_path].items()
        ranks = min(print_ranks, len(diff_scores))
        topN = sorted(diff_scores, key=lambda x: x[1])[0:ranks]
        topN = [str(name_score) for name_score in topN]
        # 比較対象が少ない場合は空文字を埋める
        for _ in range(print_ranks - len(topN)):
            topN.append('')
        topN.insert(0, img_path)
        tmp_df = pd.DataFrame([topN], columns=header, dtype='object')
        result_df = pd.concat([result_df, tmp_df])
    result_df.to_csv(output_path, header=True, index=False)


def find_img_recursive(find_dir, sep):
    img_list = []
    for ext in ['jpg', 'png', 'jpeg']:
        img_list.extend(glob.glob(find_dir + sep + '**' + sep + '*.' + ext,
                                  recursive=True))
    return img_list


def create_result_df(match_dict):
    result_df = pd.DataFrame(columns=['pair_img', 'score', 'img'])
    for img in match_dict.keys():
        tmp_df = pd.DataFrame(list(match_dict[img].items()))
        tmp_df.columns = ['pair_img', 'score']
        tmp_df['img'] = img
        result_df = pd.concat([result_df, tmp_df])
    result_df = result_df.loc[:, ['img', 'pair_img', 'score']]
    return result_df


def main():
    global logger
    global DIR_SEP
    # 入力チェック(型までは見ない)
    args = parse_args()
    config = util.get_config(args)
    if config is None:
        sys.exit(1)
    valid_input = check_config(config)
    if not valid_input:
        sys.exit(1)

    if config['mode']['debug']:
        logger = my_logger_cls.setLevelUtil(logger, DEBUG)

    SAVE_SCORE = config['output']['save_score']
    SAVE_PATH = config['output']['save_path']
    OUTPUT_PATH = config['output']['output_path']
    SCORE_PATH = config['input']['score_path']
    REGISTERED_DIR = config['input']['registered_dir']

    CLASSIFY_MULTI = config['mode']['classify_multi']
    CALC_MEAN = config['options']['calc_mean']
    PRINT_RANK = config['options']['print_rank']

    if config['options']['change_bright']:
        threthold_W = config['options']['threthold_W']
        threthold_B = config['options']['threthold_B']
    else:
        threthold_W = None
        threthold_B = None

    output_dir = os.path.dirname(OUTPUT_PATH)
    if DIR_SEP in output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if SAVE_SCORE:
        score_dir = os.path.dirname(SAVE_PATH)
        if DIR_SEP in score_dir and not os.path.exists(score_dir):
            os.makedirs(score_dir)
    if CLASSIFY_MULTI:
        CLASSIFY_DIR = config['input']['classify_dir']
        # TODO 明らかに違うディレクトリを除外するオプションができたら処理追加
        unknown_img_list = find_img_recursive(CLASSIFY_DIR, DIR_SEP)
    else:
        unknown_img_list = [config['input']['img_path']]
    if SCORE_PATH is not None:
        match_dict = read_pre_calc_score(SCORE_PATH)
    else:
        match_dict = {}

    for unknown_img in tqdm(unknown_img_list):
        check_diff_list = find_img_recursive(REGISTERED_DIR, DIR_SEP)
        get_match_points_dict = {
            other_img: get_match_points(unknown_img, other_img, match_dict,
                                        threthold_W, threthold_B)
            for other_img in tqdm(check_diff_list)
        }
        match_dict[unknown_img] = get_match_points_dict
    if DIR_SEP in output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.debug('match_dict : ' + str(match_dict), extra=extra_args)
    # result_df = create_result_df(match_dict)
    # result_df.to_csv(OUTPUT_PATH, index=False, header=True)
    write_result(match_dict, OUTPUT_PATH, CALC_MEAN, PRINT_RANK)
    if SAVE_SCORE:
        score_dir = os.path.dirname(SAVE_PATH)
        if DIR_SEP in score_dir and not os.path.exists(score_dir):
            os.makedirs(score_dir)
        dump_score(SAVE_PATH, match_dict)

    logger.debug('your inputs : ' + str(config), extra=extra_args)
    logger.debug('check_diff_list : ' + str(check_diff_list), extra=extra_args)


if __name__ == '__main__':
    main()
