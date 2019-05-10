import numpy as np
import cv2
import os
import sys

# current_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, current_path + '/../')
# from .. import util
import util

def test_expand2square_no_backcolor():
    img = np.asarray([[0, 0, 0],
                      [0, 0, 0]])
    assert (util.expand2square(img) == \
            np.asarray([[0, 0, 0],
                        [0, 0, 0],
                        [255, 255, 255]])
           ).all()
