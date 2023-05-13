import unittest

import tensorflow as tf

from zoo.blocks.rcnn import RoiPooling


class TestRCNN(unittest.TestCase):
    def test_roi_pooling(self):
        tensor = tf.reshape(tf.range(3 * 6 * 6), (1, 6, 6, 3))
        two_rois = tf.constant(
            [
                [1, 3, 1, 3],  # xmin, xmax, ymin, ymax
                [2, 5, 3, 5],  # xmin, xmax, ymin, ymax
            ]
        )
        ta = tf.TensorArray(size=2)
        ta.write(0, tensor)
        ta.write(1, two_rois)
        out = RoiPooling(2, 2).call(ta)
