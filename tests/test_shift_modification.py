import numpy as np
import argparse
import pytest
import os
import json
import cv2

import sys
sys.path.append(os.getcwd())
sys.path.append('../src/')
sys.path.append('../src/log_mod/')

import util
import shift_modification


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
