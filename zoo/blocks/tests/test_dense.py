import unittest

import tensorflow as tf

from zoo.blocks.dense import DenseBlock


class TestDenseBlock(unittest.TestCase):
    def test_creation(self):
        block = DenseBlock([(10, "relu"), (10, "sigmoid"), (5, "relu")])

        self.assertEqual(len(block._layers), 3)

        expected_activations = ("relu", "sigmoid", "relu")
        expected_names = ("fc_0_10_relu", "fc_1_10_sigmoid", "fc_2_5_relu")
        expected_units = (10, 10, 5)
        for layer, act, name, units in zip(
            block._layers, expected_activations, expected_names, expected_units
        ):
            layer_config = layer.get_config()

            self.assertEqual(
                layer_config["activation"],
                act,
            )
            self.assertEqual(
                layer_config["name"],
                name,
            )

            self.assertEqual(
                layer_config["units"],
                units,
            )

    def test_dense_block_two_dim_input(self):
        block_in = tf.zeros(
            (
                1,
                20,
            ),
            dtype=tf.dtypes.float32,
        )

        block = DenseBlock([(10, "relu"), (10, "sigmoid"), (5, "relu")])
        out_ = block(block_in)

        self.assertEqual(out_.shape, tf.TensorShape((1, 5)))


if __name__ == "__main__":
    unittest.main()
