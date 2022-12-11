"""Metrics for cifar100 task."""
from typing import final

import tensorflow as tf

from interfaces.objective import (
    Objective,
    ObjectiveFormulation,
    ObjectiveTerm,
    PerformanceMetrics,
)


@final
class Cifar100Objective(Objective):
    """Cifar100 loss and performance metrics formulation."""

    def _build_objective_formulation(self) -> ObjectiveFormulation:
        return ObjectiveFormulation(
            terms=(
                ObjectiveTerm.build(
                    name="cross_entropy_loss",
                    loss_func=tf.keras.losses.SparseCategoricalCrossentropy(
                        from_logits=True
                    ),
                    weight=1.0,
                ),
            ),
            cumulative_loss_metric=tf.keras.metrics.Mean("total_loss"),
        )

    def _build_performance_metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            tf.keras.metrics.SparseCategoricalAccuracy("accuracy"),
        )
