"""Deep neural network base."""
import abc
from typing import Dict

import tensorflow as tf


class ModelBackbone(abc.ABC, tf.keras.layers.Layer):
    """Base class for a model backbone."""


class ModelHead(abc.ABC, tf.keras.layers.Layer):
    """Base class for a model head.

    Defines the tasks of the model, therefore it is application specific.
    """


Heads = Dict[str, ModelHead]


class AssembledModel(tf.keras.Model):
    """A neural network that consists of a backbone and at least one head."""

    def __init__(
        self, backbone: ModelBackbone, heads: Heads, input_signature: tf.TensorSpec
    ):
        """Create an assembled model with a backbone and at least one head."""
        super().__init__()
        self._backbone = backbone
        self._heads = heads
        self._input_signature = input_signature

    def call(self, model_in, training) -> Dict[str, tf.Tensor]:
        """Run the model given in input."""
        backbone_output = self._backbone(model_in, training=training)
        head_outputs = {}
        for name, head in self._heads.items():
            head_outputs[name] = head(backbone_output, training=training)
        return head_outputs

    @property
    def input_signature(self) -> tf.TensorSpec:
        """Get model input signature."""
        return self._input_signature
