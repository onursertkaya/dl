import unittest

import tensorflow as tf

from zoo.blocks.resnet import ResidualUnit, ResnetBlockA, ResnetBlockB, ResnetBlockC


class TestResnetBlocks(unittest.TestCase):
    def setUp(self):
        self._name = "dummy"
        self._num_filters = 8
        self._spatial_filter_size = 10

        self._in_shape = (
            1,
            self._spatial_filter_size,
            self._spatial_filter_size,
            self._num_filters,
        )
        self._non_downsampled_out_shape = self._in_shape
        self._downsampled_out_shape = (
            1,
            self._spatial_filter_size / 2,
            self._spatial_filter_size / 2,
            self._num_filters * 2,
        )

        self._in = tf.zeros(self._in_shape, dtype=tf.dtypes.float32)

    def test_block_a_non_dowsampling(self):
        for batchnorm in [True, False]:

            block = ResnetBlockA(
                self._name,
                self._num_filters,
                use_batch_norm=batchnorm,
                downsample=False,
            )
            out_ = block(self._in)

            self.assertEqual(out_.shape, self._non_downsampled_out_shape)

    def test_block_a_dowsampling(self):
        for batchnorm in [True, False]:

            block = ResnetBlockA(
                self._name, 32, use_batch_norm=batchnorm, downsample=True
            )

            in_ = tf.zeros((1, 10, 10, 16), dtype=tf.dtypes.float32)
            out_ = block(in_)

            self.assertEqual(out_.shape, tf.TensorShape((1, 5, 5, 32)))

    def test_block_b_non_dowsampling(self):
        for batchnorm in [True, False]:

            block = ResnetBlockB(
                self._name, 32, use_batch_norm=batchnorm, downsample=False
            )

            in_ = tf.zeros((1, 10, 10, 32), dtype=tf.dtypes.float32)
            out_ = block(in_)

            self.assertEqual(out_.shape, tf.TensorShape((1, 10, 10, 32)))

    def test_block_b_dowsampling(self):
        for batchnorm in [True, False]:

            block = ResnetBlockB(
                self._name, 32, use_batch_norm=batchnorm, downsample=True
            )

            in_ = tf.zeros((1, 10, 10, 16), dtype=tf.dtypes.float32)
            out_ = block(in_)

            self.assertEqual(out_.shape, tf.TensorShape((1, 5, 5, 32)))

    def test_block_c_non_dowsampling(self):
        for batchnorm in [True, False]:

            block = ResnetBlockC(
                self._name, 32, use_batch_norm=batchnorm, downsample=False
            )

            in_ = tf.zeros((1, 10, 10, 32), dtype=tf.dtypes.float32)
            out_ = block(in_)

            self.assertEqual(out_.shape, tf.TensorShape((1, 10, 10, 32)))

    def test_block_c_dowsampling(self):
        for batchnorm in [True, False]:

            block = ResnetBlockC(
                self._name, 32, use_batch_norm=batchnorm, downsample=True
            )

            in_ = tf.zeros((1, 10, 10, 16), dtype=tf.dtypes.float32)
            out_ = block(in_)

            self.assertEqual(out_.shape, tf.TensorShape((1, 5, 5, 32)))
