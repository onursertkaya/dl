"""Deep neural network base."""
import abc

import tensorflow as tf


class ModelBackbone(abc.ABC, tf.keras.layers.Layer):
    """Base class for a model backbone."""


class ModelHead(abc.ABC, tf.keras.layers.Layer):
    """Base class for a model head.

    Defines the tasks of the model, therefore it is application specific.
    """
