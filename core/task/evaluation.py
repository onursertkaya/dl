from typing import Dict, Iterable, final

import tensorflow as tf

from core.task.task import Task
from interfaces.metrics import Metrics
from zoo.models.base import AssembledModel


@final
class Evaluation(Task):
    """Evaluation mode."""

    def __init__(self, objective: Metrics):
        """Initialize evaluation mode."""
        self._objective = objective

    @tf.function
    def step(
        self, model: AssembledModel, data_points, named_labels: Dict[str, tf.Tensor]
    ):
        head_predictions = model(data_points, training=False)
        for name, head_output in head_predictions.items():
            head_labels = named_labels[name]
            _ = self._objective(head_labels, head_output)

    def print_stats(self, epoch: int, epochs: int, batch: int, batches: int):
        def format_counter(prefix: str, current: int, total: int, sep=" / "):
            t_w = len(str(total))
            return f"{prefix}: {current:{t_w}}{sep}{total}"

        def format_metric(metrics: Iterable[tf.keras.metrics.Metric], sep=" / "):
            t_w = 8  # metrics total digits
            f_w = 6  # metrics decimal digits
            return sep.join(
                [f"{m.name}: {float(m.result()):{t_w}.{f_w}}" for m in metrics]
            )

        epoch_count = format_counter(f"{type(self).__name__} E", epoch, epochs)
        batch_count = format_counter("B", batch, batches)

        losses = format_metric(self.loss_metrics)
        perf_metrics = format_metric(self.performance_metrics)

        print(
            " | ".join(
                [
                    epoch_count,
                    batch_count,
                    losses,
                    perf_metrics,
                ]
            )
        )

    @property
    def loss_metrics(self):
        return self._objective.losses.metrics

    @property
    def performance_metrics(self):
        return self._objective.performance_metrics.metrics
