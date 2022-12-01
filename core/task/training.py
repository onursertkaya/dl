"""Training task."""
from typing import Dict, final

import tensorflow as tf

from core.task.task import Task
from interfaces.objective import Objective
from zoo.models.base import AssembledModel


@final
class Training(Task):
    """Training task."""

    def __init__(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        objective: Objective,
    ):
        """Initialize training task."""
        super().__init__(objective)
        self._optimizer = optimizer

    @tf.function
    def step(
        self, model: AssembledModel, data_points, named_labels: Dict[str, tf.Tensor]
    ):
        """One step of forward-backward pass."""
        with tf.GradientTape() as tape:
            task_losses = {}
            head_predictions = model(data_points, training=True)
            for name, head_output in head_predictions.items():
                head_labels = named_labels[name]
                task_losses[name] = self._objective(head_labels, head_output)

        gradients = tape.gradient(task_losses, model.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def reset_states(self):
        """Reset the loss and metric states."""
        for loss in self.loss_metrics:
            loss.reset_states()
        for metric in self.performance_metrics:
            metric.reset_states()
