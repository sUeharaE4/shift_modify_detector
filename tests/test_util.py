import numpy as np
import argparse
import pytest
import os
import json

import util


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


@pytest.mark.parametrize('input_csv, expect_json', [
    ('csv2json/01.csv', 'csv2json/01.json'),
])
def test_csv2json(input_csv, expect_json):
    with open(expect_json, 'r', encoding='utf-8') as j:
        expect = json.load(j)
    json_csv = util.csv2json(input_csv)
    assert json_csv == expect
