import numpy as np
import pandas as pd
import argparse
import pytest
import os
import json
import cv2
import base64

import sys
sys.path.append(os.getcwd())
sys.path.append('../src/')
sys.path.append('../src/log_mod/')

import util

TEST_TMP_DIR = 'test_tmp'
DIR_SEP = os.sep


def test_expand2square_no_backcolor():
    img = np.asarray([[0, 0, 0],
                      [0, 0, 0]])
    assert (util.expand2square(img) == \
            np.asarray([[0, 0, 0],
                        [0, 0, 0],
                        [255, 255, 255]])
           ).all()


@pytest.mark.parametrize('conf_path', [
    ('conf/detect_default.yml'),
    ('conf/shift_default.yml'),
])
def test_read_config(conf_path):
    config = util.read_config(conf_path)
    for conf_key in config.keys():
        assert config[conf_key] is not None


@pytest.mark.parametrize('conf_path, input_args', [
    ('conf/detect_default.yml', {'mode': {'debug': False}}),
    ('conf/shift_default.yml',  {'options': {'threthold_BW': 150}}),
])
def test_set_config(conf_path, input_args):
    config = util.read_config(conf_path)
    test_args = argparse.Namespace()
    conf_type = list(input_args.keys())[0]
    conf_key = list(input_args[conf_type].keys())[0]
    if conf_type == 'mode' and conf_key == 'debug':
        test_args.debug = input_args['mode']['debug']
    if conf_type == 'options' and conf_key == 'threthold_BW':
        test_args.threthold_BW = input_args['options']['threthold_BW']
    config_update = util.set_config(config, test_args)
    # 上書き確認
    assert config_update[conf_type][conf_key] == input_args[conf_type][conf_key]
    # 未指定が更新されていないことの確認
    for type_key in config.keys():
        for key in config[type_key].keys():
            if key != conf_key:
                assert config[type_key][key] == config_update[type_key][key]


@pytest.mark.parametrize('conf_path', [
    ('conf/detect_default.yml'),
    ('conf/shift_default.yml'),
    ('conf/detect_path_posix.yml'),
])
def test_modify_path_in_config(conf_path):
    conf_type_has_path = ['input', 'output']
    sep_change_dict = {'/': '\\', '\\': '/'}
    sep_dir = os.sep
    other_sep_dir = sep_change_dict[sep_dir]
    config = util.read_config(conf_path)
    config_update = util.modify_path_in_config(config)
    for conf_type in conf_type_has_path:
        for key in config_update[conf_type].keys():
            assert config_update[conf_type][key].count(other_sep_dir) == 0


@pytest.mark.parametrize('input_path', [
    ('conf/detect_default.yml'),
    ('NotFoundPath'),
])
def test_get_config(input_path):
    test_args = argparse.Namespace()
    test_args.conf_path = input_path
    config = util.get_config(test_args)
    if input_path == 'NotFoundPath':
        assert config is None
    else:
        for conf_key in config.keys():
            assert config[conf_key] is not None


@pytest.mark.parametrize('input_csv, expect_json', [
    ('csv2json/01.csv', 'csv2json/01.json'),
])
def test_csv2json(input_csv, expect_json):
    with open(expect_json, 'r', encoding='utf-8') as j:
        expect = json.load(j)
    json_csv = util.csv2json(input_csv)
    assert json_csv == expect


@pytest.mark.parametrize('input_csv, input_img, expect_json', [
    ('csv2json/01.csv', 'images/mnist_7.png', 'api_json/text_detect_req_01.json'),
])
def test_create_text_detect_request(input_csv, input_img, expect_json):
    rectangle_json = util.csv2json(input_csv)
    with open(expect_json, 'r', encoding='utf-8') as j:
        expect = json.dumps(json.load(j))
    img = cv2.imread(input_img, cv2.IMREAD_COLOR)
    api_json = util.create_text_detect_request(rectangle_json, img)
    assert api_json == expect

    api_dict = json.loads(api_json)
    img_as_text = api_dict['image']

    img_binary = base64.b64decode(img_as_text.encode('utf-8'))
    img_array = np.frombuffer(img_binary, dtype=np.uint8)

    img_from_text = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    assert type(img) == type(img_from_text)
    assert img.shape == img_from_text.shape


@pytest.mark.parametrize('input_img', [
    ('images/mnist_7.png'),
])
def test_img2float64(input_img):
    img = cv2.imread(input_img)
    float_img = util.img2float64(img)
    assert float_img.dtype == np.float64
    assert img.shape == float_img.shape


@pytest.mark.parametrize('input_dir, save_size', [
    ('input', False),
    ('input', True),
])
def test_uniform_img_size(need_work_dir, input_dir, save_size):
    util.uniform_img_size(input_dir, TEST_TMP_DIR, save_size)
    img_name_list = os.listdir(TEST_TMP_DIR)
    if save_size:
        assert 'size.csv' in img_name_list
        img_name_list.remove('size.csv')

    img_paths = [os.path.join(TEST_TMP_DIR, img_name)
                 for img_name in img_name_list]
    img_shape_list = [cv2.imread(img_path).shape for img_path in img_paths]
    img_shape_set = set(img_shape_list)
    uniformed_size = img_shape_set.pop()
    # 全サイズが同じであればsetの中身は1個。popしたから0になるはず。
    assert len(img_shape_set) == 0
    input_paths = [os.path.join(input_dir, img_name)
                   for img_name in img_name_list]
    df = pd.DataFrame([cv2.imread(img_path).shape for img_path in input_paths])
    df.columns = ['width', 'height', 'ch']
    assert uniformed_size == (df['width'].max(), df['height'].max(), 3)


def test_concat_path():
    dir_separated_list = os.getcwd().split(DIR_SEP)
    concat_result = util.concat_path(dir_separated_list, DIR_SEP)
    assert concat_result == os.getcwd()
