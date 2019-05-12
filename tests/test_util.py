import numpy as np
import argparse
import pytest

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

    config = util.set_config(config, test_args)
    assert config[conf_type][conf_key] == input_args[conf_type][conf_key]
