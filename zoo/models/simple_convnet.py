"""Simple convolutional neural network model."""
from typing import final

import tensorflow as tf

from zoo.models.base import ModelBackbone


@final
class SimpleConvNet(ModelBackbone):
    """A one-layer convolutional neural net."""

    def __init__(self, kernel_size: int, stride: int, num_filters: int):
        """Create a model."""
        super().__init__()

        self.conv = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=(
                kernel_size,
                kernel_size,
            ),
            strides=(
                stride,
                stride,
            ),
            padding="same",
            data_format="channels_last",
            activation=None,
            use_bias=True,
            name="conv",
        )

    def call(self, input_tensor):
        """Run the model given an input."""
        return self.conv(input_tensor)
