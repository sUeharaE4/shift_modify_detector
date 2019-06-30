from logging import INFO, DEBUG
import argparse
from distutils.util import strtobool
import sys
import os
from os.path import isfile
import cv2
import requests
import pandas as pd
import json

import util
from log_mod import modify_logger

# ログの追加フォーマット
extra_args = {}
extra_args['tab'] = '\t'

detect_logger_cls = modify_logger.ModifyLogger()
logger = detect_logger_cls.create_logger(__name__, INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='抽出済みの矩形から枠線等文字のない部分を除外する' +\
                                                  'スクリプト. Dockerコンテナが起動していることが必要.')
    # 入力画像と出力先
    parser.add_argument('--image_path', type=str,
                        help='画像のパス.')
    parser.add_argument('--csv_path', type=str,
                        help='矩形の座標情報を記載したCSVのパス.')
    parser.add_argument('--output_dir', type=str,
                        help='文字のない部分を除去したCSVを出力するパス.')
    # ズレ修正でディレクトリを指定するか
    # TODO 将来的にはサブコマンドで実装してほしい
    parser.add_argument('--lang', type=strtobool,
                        help='文字が日本語のみならjpn. アルファベットを含む場合はeng+jpn')
    # 比較画像を作成するか
    parser.add_argument('--create_diff', type=strtobool,
                        help='除外した矩形を画像化するかどうか.')
    # Debug log を出力するか
    parser.add_argument('--debug', type=strtobool,
                        help='debug log をコンソールに出力する場合はTrue.出力する場合進捗表示が崩れる.')
    parser.add_argument('--conf_path', type=str, default='../conf/tesseract.yml',
                        help='設定ファイルのパス.デフォルトは../conf/tesseract.yml.')

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
    image_path = config['input']['image_path']
    csv_path = config['input']['csv_path']

    if image_path is None:
        print('画像のパスを指定してください.例：--image_path ../input/0001.jpg')
        return False
    if not isfile(image_path):
        print('画像が存在しません.パスを確認してください.')
        return False
    if csv_path is None:# and not detect_multi:
        print('csvのパスを指定してください.')
        print('例1：--csv_path input/other.jpg')
        return False
    # 問題なければTrueを返却する
    return True


def main():
    global logger
    # 入力チェック(型までは見ない)
    args = parse_args()
    config = util.get_config(args)
    if config is None:
        sys.exit(1)
    valid_input = check_config(config)
    if not valid_input:
        sys.exit(1)

    if config['mode']['debug']:
        logger = detect_logger_cls.setLevelUtil(logger, DEBUG)

    health_check_url = config['url']['health_check']
    logger.debug('health_check_url : ' + str(health_check_url), extra=extra_args)
    response_health = requests.get(health_check_url)
    logger.debug('health_check_status is : ' + str(response_health.status_code), extra=extra_args)
    if response_health.status_code != 200:
        print('API サーバと疎通できませんでした.')
        sys.exit(1)

    IMAGE_PATH = config['input']['image_path']
    CSV_PATH = config['input']['csv_path']
    OUTPUT_DIR = config['output']['output_dir']
    CREATE_DIFF = config['mode']['create_diff']

    DIR_SEP = os.sep

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    rectangle_json = util.csv2json(CSV_PATH)
    logger.debug('rectangle_json : ' + str(rectangle_json), extra=extra_args)
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    req_json = util.create_text_detect_request(rectangle_json, img)
    logger.debug('type(req_json) : ' + str(type(req_json)), extra=extra_args)

    text_detect_url = config['url']['text_detect']
    logger.debug('text_detect_url : ' + text_detect_url, extra=extra_args)
    response_detect = requests.post(text_detect_url, json=req_json)
    logger.debug('response_detect : ' + str(response_detect), extra=extra_args)
    res_json = response_detect.json()
    logger.debug('response : ' + str(res_json), extra=extra_args)
    req_df = pd.DataFrame(json.loads(req_json)['rectangles'])
    logger.debug('req_df : ' + str(req_df), extra=extra_args)
    res_df = pd.DataFrame(res_json, columns=['uid', 'has_text'])
    logger.debug('res_df : ' + str(res_df), extra=extra_args)
    rectangle_df = pd.merge(req_df, res_df, on='uid')
    logger.debug('rectangle_df : ' + str(rectangle_df), extra=extra_args)
    has_text_df = rectangle_df[rectangle_df['has_text']]
    has_text_df = has_text_df.drop(res_df.columns, axis=1)
    logger.debug('has_text_df : ' + str(has_text_df), extra=extra_args)

    csv_header = pd.read_csv(CSV_PATH).columns
    has_text_df = has_text_df.loc[:, csv_header]
    has_text_df.to_csv(os.path.join(OUTPUT_DIR, CSV_PATH.split(DIR_SEP)[-1]),
                       index=False, header=True)

    if CREATE_DIFF:
        droped_df = rectangle_df[~rectangle_df['has_text']]
        droped_df = droped_df.drop(res_df.columns, axis=1)
        droped_df = droped_df.loc[:, csv_header]
        logger.debug('droped_df : ' + str(droped_df), extra=extra_args)
        for i, rect in enumerate(droped_df.values):
            x, y, width, height = rect
            clip_img = img[y:y+height, x:x+width]
            img_name = IMAGE_PATH.split(DIR_SEP)[-1].split('.')[0] + '_' + format(i, '0>4')
            cv2.imwrite(os.path.join(OUTPUT_DIR, img_name + '.jpg'), clip_img)

    logger.debug('your inputs : ' + str(config), extra=extra_args)


if __name__ == '__main__':
    main()
