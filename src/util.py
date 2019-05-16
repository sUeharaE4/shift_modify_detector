import numpy as np
import cv2
import yaml
import os
import pandas as pd
import base64


def expand2square(img, background_color=None):
    """
    画像が正方形になるように画素を追加して拡張する.
    Parameters
    ----------
    img : numpy.ndarray
        正方形にしたい画像.
    background_color : tuple
        追加する画素の背景色。指定がなければ白.

    Returns
    -------
    expand_img : numpy.ndarray
        正方形にした画像

    """
    height, width = img.shape[:2]
    if background_color is None:
        background_color = [255 for _ in range(len(img.shape))]
    expand_img = img.copy()
    # とりあえず拡張する画素数を0として、縦横で大きさに差があれば拡張する
    expand_top, expand_bottom, expand_left, expand_right = 0, 0, 0, 0
    if width > height:
        expand_bottom = width - height
    if height > width:
        expand_right = height - width
    expand_img = cv2.copyMakeBorder(expand_img, expand_top, expand_bottom, expand_left, expand_right,
                                    cv2.BORDER_CONSTANT, value=background_color)
    return expand_img


def expand_power2(img, background_color=None):
    """
    正方形画像の辺が2の累乗になるように画素を追加して拡張する.
    Parameters
    ----------
    img : numpy.ndarray
        拡張したい画像.
    background_color : tuple
        追加する画素の背景色。指定がなければ白.

    Returns
    -------
    expand_img : numpy.ndarray
        1辺の長さが2の累乗にした画像
    """
    height, width = img.shape[:2]
    assert height == width, '先に正方形にしてください'
    if background_color is None:
        background_color = [255 for _ in range(len(img.shape))]
    expand_img = img.copy()
    expand_top, expand_bottom, expand_left, expand_right = 0, 0, 0, 0
    expand_power = 1
    while height > 2**expand_power:
        expand_power = expand_power+1
    expand_bottom, expand_right = 2**expand_power - height, 2**expand_power - width
    expand_img = cv2.copyMakeBorder(expand_img, expand_top, expand_bottom, expand_left, expand_right,
                                    cv2.BORDER_CONSTANT, value=background_color)
    return expand_img


def expand_cut2base_size(base_img, written_img, background_color=None):
    """
    base_imgに合わせて画像をパディングか切り出しする
    いずれも右側、下側に追加or削除を実施する.
    Parameters
    ----------
    base_img : numpy.ndarray
        サイズ変更する基準サイズを与える画像.
    written_img : numpy.ndarray
        サイズ変更される画像.
    background_color : tuple
        追加する画素の背景色。指定がなければ白.

    Returns
    -------
    modify_written : numpy.ndarray
        base_imgにサイズを合わせられた入力画像.
    """
    def modify_img(img, diff_width, diff_height, background_color):
        height, width = img.shape[:2]
        modify_img = img.copy()
        if diff_width < 0:
            modify_img = modify_img[:, :width - diff_width]
        if diff_height < 0:
            modify_img = modify_img[:height - diff_height, :]
        if diff_width > 0:
            modify_img = cv2.copyMakeBorder(modify_img, 0, 0, 0, diff_width,
                                            cv2.BORDER_CONSTANT, value=background_color)
        if diff_width > 0:
            modify_img = cv2.copyMakeBorder(modify_img, 0, diff_height, 0, 0,
                                            cv2.BORDER_CONSTANT, value=background_color)
        return modify_img

    if base_img.shape == written_img.shape:
        return written_img
    if background_color is None:
        background_color = [255 for _ in range(len(base_img.shape))]
    base_height, base_width = base_img.shape[:2]
    written_height, written_width = written_img.shape[:2]
    diff_height = base_height - written_height
    diff_width = base_width - written_width
    modify_written = modify_img(written_img, diff_width, diff_height, background_color)

    return modify_written


def exchange_black_white(img):
    """
    画像の白黒を反転する.
    Parameters
    ----------
    img : numpy.ndarray
        白黒反転したい画像.

    Returns
    -------
    numpy.ndarray
        白黒反転した画像.

    """
    return 255 - img


def calc_mask(base_img, written_img, threshold=230, with_dilation=False, kernel=np.ones((1, 1), np.uint8), itr=1):
    """

    Parameters
    ----------
    in_img1 : numpy.ndarray
        基準となる画像.
    written_img : numpy.ndarray
        差分を抽出したい画像.
    threshold : int
        差分を2値化する際の閾値. 0～255.
    with_dilation : bool
        差分を抽出する時にdilationしておくか. 画像にズレがある場合はTrue
    kernel : numpy.ndarray
        dilationする際のkernel. (N,N)行列.
    itr : int
        dilationする際のiteration回数.

    Returns
    -------
    mask : numpy.ndarray
        差分を抽出する際のマスク

    """
    assert base_img.shape == written_img.shape, '画像のサイズが異なります'
    img1 = base_img.copy()
    img2 = written_img.copy()
    if with_dilation:
        img1 = cv2.dilate(img1, kernel, iterations=itr)
        img2 = cv2.dilate(img2, kernel, iterations=itr)
    # 差分の絶対値を計算
    mask = cv2.absdiff(img1, img2)
    # 差分画像を二値化してマスク画像を算出
    mask[mask < threshold] = 0
    mask[mask >= threshold] = 255
    if with_dilation:
        mask = cv2.erode(mask, kernel, iterations=itr)
    return mask


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
    base_img = exchange_black_white(cv2.imread(base_img_path, 0))
    pair_img = exchange_black_white(cv2.imread(pair_img_path, 0))
    # baseとpairのサイズを合わせる
    pair_img = expand_cut2base_size(base_img, pair_img)
    # 2値化
    for img in [base_img, pair_img]:
        __binarize(img, threshold)
    return [base_img, pair_img]


def expand_imgs(base_img, pair_img, background_color=[0, 0]):
    base_img = expand2square(base_img, background_color)
    pair_img = expand2square(pair_img, background_color)
    return [base_img, pair_img]


def resize_imgs(base_img, pair_img, resize_shape):
    base_img_resize = cv2.resize(base_img, resize_shape)
    pair_img_resize = cv2.resize(pair_img, resize_shape)
    return [base_img_resize, pair_img_resize]


def read_config(conf_path):
    """
    設定ファイルを読み込みdict形式で返却する.
    Parameters
    ----------
    conf_path : str
        設定ファイルのパス

    Returns
    -------
    config : dict
        読み込んだ設定値
    """
    with open(conf_path, 'r', encoding='utf-8') as yml:
        config = yaml.load(yml, Loader=yaml.SafeLoader)
    return config


def set_config(config, args):
    """
    実行時引数で指定されたパラメタでconfigを読み込んだオブジェクトを更新する.
    Parameters
    ----------
    config : dict
        設定ファイルを読み込んだオブジェクト
    args : Namespace
        実行時引数をparseした結果

    Returns
    -------
    config : dict
        更新した設定値
    """
    for key, value in sorted(vars(args).items()):
        # デフォルト値はNoneなので、Noneは無視.
        if value is not None:
            for conf_type in config.keys():
                if key in config[conf_type]:
                    config[conf_type][key] = value
    return config


def modify_path_in_config(config):
    """
    configのパスをOSに合わせて変更する.
    Parameters
    ----------
    config : dict
        設定ファイルを読み込んだオブジェクト

    Returns
    -------
    config : dict
        更新した設定値
    """
    conf_type_has_path = ['input', 'output']
    sep_change_dict = {'/': '\\', '\\': '/'}
    sep_dir = os.sep
    other_sep_dir = sep_change_dict[sep_dir]
    for conf_type in conf_type_has_path:
        for key in config[conf_type].keys():
            config[conf_type][key] = config[conf_type][key].replace(other_sep_dir, sep_dir)
    return config


def csv2json(csv_path):
    """
    CSVの矩形座標情報をJSONに変換する.
    Parameters
    ----------
    csv_path : str
        CSVファイルのパス

    Returns
    -------
    rect_json : dict
        矩形情報を格納したJSON
    """
    df = pd.read_csv(csv_path)
    rect_json = {'rectangles': df.to_dict(orient='records')}
    return rect_json


def create_text_detect_request(rectangle_json, img):
    """
    TesseractのTextDetection用jsonを生成する.
    Parameters
    ----------
    rectangle_json : dict
        座標情報を格納したJSON.
    img : numpy.ndarray
        画像.

    Returns
    -------
    api_json : dict
        api request のJSON. 画像の追加と座標情報にIDが追加されている.
    """
    for uid, rect in enumerate(rectangle_json['rectangles']):
        rect['uid'] = uid
    base64_text = base64.b64encode(img).decode('utf-8')
    api_json = {'image': base64_text}
    api_json.update(rectangle_json)
    return api_json
