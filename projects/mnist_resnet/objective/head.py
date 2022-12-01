"""Mnist head implementation."""
import tensorflow as tf

from zoo.blocks.dense import DenseBlock
from zoo.models.base import ModelHead


class MnistDenseHead(ModelHead):
    """Densely connected head for mnist task."""

    def __init__(self):
        """Create a head."""
        super().__init__()
        self._flatten = tf.keras.layers.Flatten()
        self._block = DenseBlock([(1000, "relu"), (10, None)])

    def call(self, layer_in, training):
        """Run the head given an input."""
        return self._block(self._flatten(layer_in), training=training)


class MnistGlobalPoolingHead(ModelHead):
    """Global average pooling head for mnist task."""

    def __init__(self):
        """Create a head."""
        super().__init__()
        self._conv = tf.keras.layers.Conv2D(
            filters=10,
            kernel_size=(
                3,
                3,
            ),
            strides=(
                1,
                1,
            ),
            padding="same",
            data_format="channels_last",
            activation=tf.keras.activations.relu,
            use_bias=True,
            name="conv",
        )
        self._global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, layer_in):
        """Run the head given an input."""
        return self._global_avg_pooling(self._conv(layer_in))
