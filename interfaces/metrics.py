import abc
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf


@dataclass(frozen=True)
class WeightedMetrics:
    metrics: Tuple[tf.keras.metrics.Metric]
    weights: Tuple[float]


@dataclass(frozen=True)
class ObjectiveFormulation(WeightedMetrics):
    funcs: Tuple[tf.keras.losses.Loss]


class Metrics(abc.ABC):
    """Interface class for losses and performance metrics."""

    @abc.abstractmethod
    def loss_factory(self) -> ObjectiveFormulation:
        pass

    @abc.abstractmethod
    def performance_metrics_factory(self) -> WeightedMetrics:
        pass

    def __init__(self):

        self._losses = self.loss_factory()
        self._performance_metrics = self.performance_metrics_factory()

        self._loss_weights = self._normalize(self._losses.weights)
        self._performance_metrics_weights = self._normalize(
            self._performance_metrics.weights
        )

    @tf.function
    def __call__(self, labels, predictions):
        """Calculate the cumulative loss value of labels and predictions."""
        losses = [loss_fn(labels, predictions) for loss_fn in self._losses.funcs]

        # populate loss values
        for loss, metric in zip(losses, self._losses.metrics):
            metric(loss)

        # populate performance metric values
        for perf_met in self._performance_metrics.metrics:
            perf_met(labels, predictions)

        cumulative_weighted_loss = tf.reduce_sum(
            tf.math.multiply(
                losses,
                self._loss_weights,
            )
        )
        return cumulative_weighted_loss

    @property
    def losses(self) -> ObjectiveFormulation:
        return self._losses

    @property
    def performance_metrics(self) -> WeightedMetrics:
        return self._performance_metrics

    @staticmethod
    def _normalize(weights: Tuple[float]) -> tf.constant:
        return tf.constant(
            tf.divide(weights, tf.reduce_sum(weights)),
            dtype=tf.float32,
        )
