from typing import final

import tensorflow as tf

from interfaces.metrics import Metrics, ObjectiveFormulation, WeightedMetrics


@final
class MnistLoss(Metrics):
    """Mnist loss and performance metrics formulation."""

    def loss_factory(self) -> ObjectiveFormulation:
        # todo: find a fucking way to assert from_logits=True when input is logits,
        # and False otherwise. lost 3 hours.
        return ObjectiveFormulation(
            funcs=(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),),
            metrics=(tf.keras.metrics.Mean("cross_entropy_loss"),),
            weights=(1.0,),
        )

    def performance_metrics_factory(self) -> WeightedMetrics:
        # merge these, maybe? mnist is a restricting experiment, maybe other projects
        # would require them separate...
        return WeightedMetrics(
            metrics=(tf.keras.metrics.SparseCategoricalAccuracy("accuracy"),),
            weights=(1.0,),
        )
