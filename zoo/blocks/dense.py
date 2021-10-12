from math import isclose
from typing import List, Tuple, Union

import tensorflow as tf

NumUnitsAndActivation = Tuple[int, Union[str, None]]
DenseBlockConfig = List[NumUnitsAndActivation]


class DenseBlock(tf.keras.layers.Layer):
    """A sequence of fully-connected layers."""

    layer_name_tpl = "fc_{idx}_{units}_{act}"

    def __init__(self, configuration: DenseBlockConfig, dropout_rate=0.0):
        super().__init__()
        self._layers = []
        self._dropouts = []

        for idx, layer_config in enumerate(configuration):
            num_units, activation = layer_config
            # assert activation in [relu, softmax, None]
            self._layers.append(
                tf.keras.layers.Dense(
                    units=num_units,
                    activation=activation,
                    name=DenseBlock.layer_name_tpl.format(
                        idx=idx, units=num_units, act=activation
                    ),
                )
            )
            self._dropouts.append(
                tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{idx}")
                if not isclose(dropout_rate, 0.0)
                else lambda x, training: x
            )

    def call(self, layer_in, training):
        """Forward pass."""
        t = layer_in
        for layer, dropout in zip(self._layers, self._dropouts):
            t = layer(t)
            t = dropout(t, training=training)

        return t
