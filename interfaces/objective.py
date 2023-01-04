"""Metrics base."""
import abc
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf


class PerformanceMetrics(tuple):
    """A set of performance metrics.

    NOTE:
    As python versions older than 3.9 does not support using
    tuple as a generic type hint, we still need to use typing.Tuple.
    Then the following does not work as an initializer:
        PerformanceMetrics = Tuple[tf.keras.metrics.Metric, ...]
    Therefore, this class is a wrapper around tuple, working both
    as a type hint and a type, until python version is updated.
    """

    def __new__(cls, *args: tf.keras.metrics.Metric):
        """Construct an instance."""
        assert all(isinstance(arg, tf.keras.metrics.Metric) for arg in args)
        return args


@dataclass(frozen=True)
class ObjectiveTerm:
    """A loss term in the loss formulation."""

    loss_func: tf.keras.losses.Loss
    loss_metric: tf.keras.metrics.Mean
    weight: float

    @classmethod
    def build(cls, name: str, loss_func: tf.keras.losses.Loss, weight: float = 1.0):
        """Create an objective term."""
        return cls(
            loss_func=loss_func, loss_metric=tf.keras.metrics.Mean(name), weight=weight
        )


@dataclass(frozen=True)
class ObjectiveFormulation:
    """A neural network loss formulation."""

    terms: Tuple[ObjectiveTerm]
    cumulative_loss_metric: tf.keras.metrics.Mean

    def calculate(self, labels, predictions):
        """Calculate the loss value given labels and predictions."""
        losses = []
        for objective_term in self.terms:
            loss = objective_term.loss_func(labels, predictions)
            objective_term.loss_metric(loss)
            losses.append(loss)

        cumulative_loss = tf.reduce_sum(
            tf.math.multiply(
                losses,
                [term.weight for term in self.terms],
            )
        )
        self.cumulative_loss_metric(cumulative_loss)
        return cumulative_loss


class Objective(abc.ABC):
    """Interface class for losses and performance metrics."""

    def __init__(self):
        """Create an objective."""
        self._objective_formulation = self._build_objective_formulation()
        self._performance_metrics = self._build_performance_metrics()

    def __call__(self, labels, predictions):
        """Calculate the cumulative loss value of labels and predictions."""
        cumulative_loss = self._objective_formulation.calculate(labels, predictions)

        for metric in self._performance_metrics:
            metric(labels, predictions)

        return cumulative_loss

    @property
    def formulation(self) -> ObjectiveFormulation:
        """Get the loss formulation."""
        return self._objective_formulation

    @property
    def performance_metrics(self) -> PerformanceMetrics:
        """Get the performance metric."""
        return self._performance_metrics

    @abc.abstractmethod
    def _build_objective_formulation(self) -> ObjectiveFormulation:
        pass

    @abc.abstractmethod
    def _build_performance_metrics(self) -> PerformanceMetrics:
        pass
