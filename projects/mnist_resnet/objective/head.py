import tensorflow as tf

from zoo.blocks.dense import DenseBlock
from zoo.models.base import ModelHead


class MnistDenseHead(ModelHead):
    def __init__(self):
        super().__init__()
        self._flatten = tf.keras.layers.Flatten()
        self._block = DenseBlock([(1000, "relu"), (10, None)])

    def call(self, layer_in, training):
        return self._block(self._flatten(layer_in), training=training)
