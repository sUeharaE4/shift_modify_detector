import pytest
from logging import INFO, DEBUG
from log_mod import modify_logger


@pytest.mark.parametrize('log_level', [
    (INFO),
    (DEBUG),
])
def test_setLevelUtil(log_level):
    modify_logger_cls = modify_logger.ModifyLogger()
    logger = modify_logger_cls.create_logger(__name__, INFO)
    logger = modify_logger_cls.setLevelUtil(logger, log_level)
    assert logger.level == log_level
    for handler in logger.handlers:
        assert handler.level == log_level


@pytest.mark.parametrize('log_level', [
    (INFO),
    (DEBUG),
])
def test_create_logger(log_level):
    modify_logger_cls = modify_logger.ModifyLogger()
    logger = modify_logger_cls.create_logger(__name__, log_level)
    assert logger.level == log_level
    for handler in logger.handlers:
        assert handler.level == log_level
