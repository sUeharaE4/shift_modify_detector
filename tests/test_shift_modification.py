import argparse
import pytest
import os
import cv2
import sys
sys.path.append(os.getcwd())
sys.path.append('../src/')
sys.path.append('../src/log_mod/')

import util
import shift_modification

TEST_TMP_DIR = 'test_tmp'
DIR_SEP = os.sep


@pytest.mark.parametrize('conf_path, input_args', [
    ('conf/shift_template_None.yml', {'template_path': None}),
    ('conf/shift_default.yml',  {'template_path': 'XXX.jpg'}),
    ('conf/shift_pair_None.yml', {'pair_path': None, 'modify_multi': False}),
    ('conf/shift_default.yml', {'modify_dir': 'XXX/', 'modify_multi': True}),
    ('conf/shift_default.yml', {}),
])
def test_check_config(conf_path, input_args):
    args = argparse.Namespace()
    args.conf_path = conf_path
    config = util.get_config(args)
    test_args = argparse.Namespace()
    test_args.__dict__ = input_args
    config_update = util.set_config(config, test_args)
    if len(input_args) > 0:
        assert not shift_modification.check_config(config_update)
    else:
        assert shift_modification.check_config(config_update)


@pytest.mark.parametrize('conf_path, input_args', [
    ('conf/shift_default.yml',  {}),
    ('conf/shift_default.yml', {'modify_multi': False}),
    ('conf/shift_default.yml', {'create_diff': False}),
])
def test_main(mocker, conf_path, input_args):
    args = argparse.Namespace()
    args.__dict__ = input_args
    args.conf_path = conf_path
    args.output_dir = TEST_TMP_DIR
    args.debug = False
    config = util.get_config(args)
    # 実行時引数を渡す代わりに上記argsを渡す
    mocker.patch('shift_modification.parse_args').return_value = args
    shift_modification.main()
    output_dir = config['output']['output_dir']
    if not config['mode']['modify_multi']:
        mod_path = config['input']['pair_path']
        img_name = mod_path.split(DIR_SEP)[-1]
        expect_path = os.path.join('shift_expect', img_name)
        test_img = cv2.imread(mod_path)
        expect_img = cv2.imread(expect_path)
        assert test_img.shape == expect_img.shape
        assert test_img.mean() != 255
    mod_img_list = os.listdir(output_dir)
    diff_img_list = []
    for i, mod_img in enumerate(mod_img_list):
        if 'diff' in mod_img:
            diff_img = mod_img_list.pop(i)
            diff_img_list.append(diff_img)
    if config['mode']['create_diff']:
        assert len(diff_img_list) > 0
    else:
        assert len(diff_img_list) == 0
    for mod_img_name in mod_img_list:
        img_name = mod_img_name.split(DIR_SEP)[-1]
        mod_path = os.path.join(output_dir, img_name)
        expect_path = os.path.join('shift_expect', img_name)
        test_img = cv2.imread(mod_path)
        expect_img = cv2.imread(expect_path)
        assert test_img.shape == expect_img.shape
        assert test_img.mean() != 255
