from typing import Dict

import tensorflow as tf


class ModelBackbone(tf.keras.layers.Layer):
    """Base class for a model backbone."""

    def __init__(self):
        super().__init__()

    def call(self):
        raise RuntimeError


class ModelHead(tf.keras.layers.Layer):
    """Base class for a model head.

    Defines the tasks of the model, therefore it is application specific.
    """

    def __init__(self):
        super().__init__()

    def call(self):
        raise RuntimeError


Heads = Dict[str, ModelHead]


class AssembledModel(tf.keras.Model):
    """A neural network that consists of a backbone and head(s)."""

    def __init__(
        self, backbone: ModelBackbone, heads: Heads, input_signature: tf.TensorSpec
    ):
        super().__init__()
        self._backbone = backbone
        self._heads = heads
        self._input_signature = input_signature

    def call(self, model_in, training) -> Dict[str, tf.Tensor]:
        backbone_output = self._backbone(model_in, training=training)
        head_outputs = {}
        for name, head in self._heads.items():
            head_outputs[name] = head(backbone_output, training=training)
        return head_outputs

    @property
    def input_signature(self) -> tf.TensorSpec:
        return self._input_signature
