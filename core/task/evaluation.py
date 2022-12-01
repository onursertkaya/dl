"""Evaluation task."""
from typing import Dict, final

import tensorflow as tf

from core.task.task import Task
from zoo.models.base import AssembledModel


@final
class Evaluation(Task):
    """Evaluation task."""

    @tf.function
    def step(
        self, model: AssembledModel, data_points, named_labels: Dict[str, tf.Tensor]
    ):
        """One step of forward pass."""
        head_predictions = model(data_points, training=False)
        for name, head_output in head_predictions.items():
            head_labels = named_labels[name]
            _ = self._objective(head_labels, head_output)
