"""Global pooling head."""
import tensorflow as tf

from interfaces.model import ModelHead


class GlobalPoolingHead(ModelHead):
    """Global average pooling head."""

    def __init__(self, num_classes: int):
        """Create a head."""
        super().__init__()
        self._conv = tf.keras.layers.Conv2D(
            filters=num_classes,
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
