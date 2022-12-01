"""Tests for resnet blocks."""
import unittest

import tensorflow as tf

from zoo.blocks.residual import ResnetBlockA, ResnetBlockB, ResnetBlockC


class TestResnetBlocks(unittest.TestCase):
    """Test all resnet blocks."""

    def setUp(self):
        """Create the test environment."""
        self._name = "dummy"
        self._num_filters = 64
        self._downsampling_layer_input = tf.zeros((1, 4, 4, 32), dtype=tf.float32)
        self._downsampling_layer_output_expected_shape = tf.TensorShape((1, 2, 2, 64))

        self._regular_layer_input = tf.zeros((1, 4, 4, 64), dtype=tf.float32)
        self._regular_layer_output_expected_shape = tf.TensorShape((1, 4, 4, 64))

    def test_wrong_num_filters_raises(self):
        """Test if an incompatible number of filters cause an exception."""
        with self.assertRaises(AssertionError):
            ResnetBlockA(
                self._name,
                8,
                use_batch_norm=False,
                downsample=False,
            )

    def test_shapes_blocks_abc(self):
        """Test all blocks with all possible configurations."""
        for variant in [ResnetBlockA, ResnetBlockB, ResnetBlockC]:
            for downsampling in [True, False]:
                for batchnorm in [True, False]:
                    block = variant(
                        self._name,
                        self._num_filters,
                        downsample=downsampling,
                        use_batch_norm=batchnorm,
                    )
                    layer_in = (
                        self._downsampling_layer_input
                        if downsampling
                        else self._regular_layer_input
                    )
                    out_expected_shape = (
                        self._downsampling_layer_output_expected_shape
                        if downsampling
                        else self._regular_layer_output_expected_shape
                    )
                    out = block(layer_in)
                    self.assertEqual(
                        out_expected_shape,
                        out.shape,
                        f"{variant.__name__} failed on "
                        "downsampling={downsampling}, "
                        "batchnorm={batchnorm}",
                    )
