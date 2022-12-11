"""Dense head implementation."""
import tensorflow as tf

from interfaces.model import ModelHead
from zoo.blocks.dense import DenseBlock, DenseBlockConfig


class DenseHead(ModelHead):
    """Densely connected head."""

    def __init__(self, config: DenseBlockConfig):
        """Create a head."""
        super().__init__()
        self._flatten = tf.keras.layers.Flatten()
        self._block = DenseBlock(config)

    def call(self, layer_in, training):
        """Run the head given an input."""
        return self._block(self._flatten(layer_in), training=training)
