import numpy as np
import argparse
import pytest
import os
import json
import cv2
import time
import sys
import pandas as pd
sys.path.append(os.getcwd())
sys.path.append('../src/')
sys.path.append('../src/log_mod/')
sys.path.append('../src/image_classifer/')

import util
import classify_img

TEST_TMP_DIR = 'test_tmp'
DIR_SEP = os.sep


@pytest.mark.parametrize('conf_path, input_args', [
    ('conf/classify_default.yml', {'img_path': None, 'classify_multi': False}),
    ('conf/classify_default.yml',  {'score_path': 'XXX'}),
    ('conf/classify_default.yml', {'classify_multi': True, 'classify_dir': 'XXX'}),
    ('conf/classify_default.yml', {'change_bright': True, 'threthold_W': None}),
    ('conf/classify_default.yml', {'change_bright': True, 'threthold_B': None}),
    ('conf/classify_default.yml', {}),
])
def test_check_config(conf_path, input_args):
    args = argparse.Namespace()
    args.__dict__ = input_args
    args.conf_path = conf_path
    config = util.get_config(args)
    if len(input_args) > 0:
        assert not classify_img.check_config(config)
    else:
        assert classify_img.check_config(config)


@pytest.mark.parametrize('unknown_path, known_path, match_dict, expect', [
    ('input/0001.jpg', 'input/template.jpg', {'input/0001.jpg': {'input/template.jpg': 3.14}}, 3.14),
    ('input/0001.jpg', 'input/0001.jpg', {'input/0001.jpg': {'input/template.jpg': 3.14}}, 0.0),
    ('input/0001.jpg', 'images/mnist_7.png', {}, 10000000),
    ('input/0001.jpg', 'input/0001.jpg', {}, 0.0),
])
def test_get_match_points(unknown_path, known_path, match_dict, expect):
    assert classify_img.get_match_points(unknown_path, known_path, match_dict) == expect


@pytest.mark.parametrize('match_dict, expect_csv_path', [
    ({'input/0001.jpg': {'input/template.jpg': 3.14},
      'input/XXX.jpg': {'input/template.jpg': 2.718}}, 'image_classify/result.csv'),
])
def test_create_result_df(match_dict, expect_csv_path):
    expect_df = pd.read_csv(expect_csv_path).reset_index(drop=True)
    result_df = classify_img.create_result_df(match_dict).reset_index(drop=True)
    assert (result_df == expect_df).all().all()


@pytest.mark.parametrize('score_dict', [
    ({'A'+DIR_SEP+'001.jpg':
        {'A'+DIR_SEP+'002.jpg': 3.14, 'B'+DIR_SEP+'001.jpg': 0.89, 'B'+DIR_SEP+'002.jpg': 0.11},
      'A'+DIR_SEP+'002.jpg':
        {'A'+DIR_SEP+'001.jpg': 3.14, 'B'+DIR_SEP+'001.jpg': 0.2, 'B'+DIR_SEP+'002.jpg': 0.3},
      'B'+DIR_SEP+'001.jpg':
        {'A'+DIR_SEP+'001.jpg': 0.89, 'A'+DIR_SEP+'002.jpg': 0.2, 'B'+DIR_SEP+'002.jpg': 0.11},
      'B'+DIR_SEP+'002.jpg':
        {'A'+DIR_SEP+'001.jpg': 0.11, 'A'+DIR_SEP+'002.jpg': 0.3, 'B'+DIR_SEP+'001.jpg': 0.11}}),
])
def test_calc_mean(score_dict):
    mean_dict = classify_img.calc_mean(score_dict)
    expect = {'A'+DIR_SEP+'001.jpg': {'A': 3.14, 'B': 0.5},
              'A'+DIR_SEP+'002.jpg': {'A': 3.14, 'B': 0.25},
              'B'+DIR_SEP+'001.jpg': {'A': 0.545, 'B': 0.11},
              'B'+DIR_SEP+'002.jpg': {'A': 0.205, 'B': 0.11},
              }
    assert mean_dict == expect
