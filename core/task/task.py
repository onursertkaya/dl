"""Task base."""
import abc
from typing import Dict

import tensorflow as tf

from interfaces.objective import Objective
from zoo.models.base import AssembledModel


class Task(abc.ABC):
    """Base class for tasks."""

    def __init__(self, objective: Objective):
        """Initialize task."""
        self._objective = objective

    @abc.abstractmethod
    def step(
        self, model: AssembledModel, data_points, named_labels: Dict[str, tf.Tensor]
    ):
        """One step of the task."""

    @property
    def loss_metrics(self):
        """Get loss metrics."""
        return [term.loss_metric for term in self._objective.formulation.terms] + [
            self._objective.formulation.cumulative_loss_metric
        ]

    @property
    def performance_metrics(self):
        """Get performance metrics."""
        return self._objective.performance_metrics
