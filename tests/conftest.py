import os
import shutil
import pytest
TEST_TMP_DIR = 'test_tmp'


@pytest.fixture(scope='function', autouse=True)
def need_work_dir():
    if not os.path.exists(TEST_TMP_DIR):
        os.mkdir(TEST_TMP_DIR)
    yield
    shutil.rmtree(TEST_TMP_DIR)
