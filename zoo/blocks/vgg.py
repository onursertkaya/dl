"""VGGNet blocks' implementation.
https://arxiv.org/abs/1409.1556
"""
import tensorflow as tf

DATA_FORMAT = "channels_last"  # NHWC
PADDING_SAME = "SAME"


class VGGNetBlock(tf.keras.layers.Layer):
    def __init__(self, level: int):
        assert 1 <= level <= 5
        level = min(level, 4)
        name = f"conv{level}"
        super().__init__(name=name)
        num_filters = 2 ** (level + 5)
        self._conv = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            activation=None,
            use_bias=True,
            name=name,
        )

    def call(self, input_tensor, training):
        """Forward pass."""
        return tf.nn.relu(self._conv(input_tensor))


class VGGNetPoolBlock(tf.keras.layers.Layer):
    def __init__(self, level: int):
        """Initialize layers."""
        super().__init__(name="initial_maxpool")
        self._maxpool = tf.keras.layers.MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding=PADDING_SAME,
            data_format=DATA_FORMAT,
            name=f"maxpool{level}",
        )

    def call(self, input_tensor, training):
        """Forward pass."""
        return self._maxpool(input_tensor)
