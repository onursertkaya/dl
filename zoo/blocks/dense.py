"""Densely connected block."""
from typing import List, Optional, Tuple

import tensorflow as tf

NumUnitsAndActivation = Tuple[int, Optional[str]]
DenseBlockConfig = List[NumUnitsAndActivation]


class DenseBlock(tf.keras.layers.Layer):
    """A sequence of fully-connected layers."""

    layer_name_tpl = "fc_{idx}_{units}_{act}"

    def __init__(self, configuration: DenseBlockConfig, dropout_rate=0.0):
        """Create a block."""
        super().__init__()
        self._layers = []
        self._dropouts = []

        for i, layer_config in enumerate(configuration):
            num_units, activation = layer_config
            # TODO: assert activation in [relu, softmax, None]
            self._layers.append(
                tf.keras.layers.Dense(
                    units=num_units,
                    activation=activation,
                    name=DenseBlock.layer_name_tpl.format(
                        idx=i, units=num_units, act=activation
                    ),
                )
            )
            self._dropouts.append(
                lambda x, training: x
                if dropout_rate == 0.0
                else tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i}")
            )

    def call(self, layer_in, training):
        """Run the block given an input."""
        intermediate = layer_in
        for layer, dropout in zip(self._layers, self._dropouts):
            intermediate = layer(intermediate)
            intermediate = dropout(intermediate, training=training)

        return intermediate
