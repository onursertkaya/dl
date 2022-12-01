"""Experiment summary."""
from typing import Iterable

import tensorflow as tf

from core.experiment_settings import ExperimentSettings
from core.task.task import Task


class Summary:
    """Experiment summary utilities."""

    def __init__(self, settings: ExperimentSettings):
        """Create a summary writer."""
        self._settings = settings
        self._stdout_writer = _StdoutWriter()
        self._tensorboard_writer = _TensorboardWriter(logdir=self._settings.directory)

    def write(
        self, task: Task, task_num_batches: int, current_batch: int, current_epoch: int
    ):
        """Write summary based on the current state of task."""
        self._print_to_stdout(
            task,
            current_batch,
            task_num_batches,
            current_epoch,
        )

        self._write_to_disk(task, current_batch, task_num_batches)

    def _print_to_stdout(
        self, task: Task, batch_ctr: int, num_total_task_batches: int, epoch_ctr
    ):
        is_modulo = self._check_modulo(
            batch_ctr, num_total_task_batches, self._settings.prints_per_epoch
        )
        is_first_or_last_batch = batch_ctr in [num_total_task_batches, 1]
        if is_modulo or is_first_or_last_batch:
            self._stdout_writer.print(
                task,
                epoch_ctr,
                self._settings.epochs,
                batch_ctr,
                num_total_task_batches,
            )

    def _write_to_disk(self, task: Task, batch_ctr: int, num_total_task_batches: int):
        if self._check_modulo(
            batch_ctr,
            num_total_task_batches,
            self._settings.disk_writes_per_epoch,
        ):
            self._tensorboard_writer.write(task)

    @staticmethod
    def _check_modulo(
        batch_ctr: int, num_total_task_batches: int, frequency: int
    ) -> bool:
        return batch_ctr % (num_total_task_batches // frequency) == 0


class _TensorboardWriter:
    """Tensorboard summary utilities."""

    def __init__(self, logdir: str):
        """Create a writer."""
        self._write_ctr = 0
        self._writer = tf.summary.create_file_writer(logdir + "/" + "tboard")

    def write(self, task: Task):
        """Write the tracked metrics of the task to the board."""
        with self._writer.as_default():
            with tf.name_scope(task.__class__.__name__):
                for loss in task.loss_metrics:
                    tf.summary.scalar(loss.name, loss.result(), step=self._write_ctr)
                for perf_metric in task.performance_metrics:
                    tf.summary.scalar(
                        perf_metric.name, perf_metric.result(), step=self._write_ctr
                    )
                self._write_ctr += 1


class _StdoutWriter:
    """Stdout summary utilities."""

    # pylint: disable=too-many-arguments
    def print(self, task: Task, epoch: int, epochs: int, batch: int, batches: int):
        """Print to stdout."""
        epoch_count = self._format_counter(f"{type(task).__name__} E", epoch, epochs)
        batch_count = self._format_counter("B", batch, batches)

        losses = self._format_metric(task.loss_metrics)
        perf_metrics = self._format_metric(task.performance_metrics)

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

    @staticmethod
    def _format_counter(prefix: str, current: int, total: int, sep=" / "):
        t_w = len(str(total))
        return f"{prefix}: {current:{t_w}}{sep}{total}"

    @staticmethod
    def _format_metric(metrics: Iterable[tf.keras.metrics.Metric], sep=" / "):
        t_w = 8  # metrics total digits
        f_w = 6  # metrics decimal digits
        return sep.join([f"{m.name}: {float(m.result()):{t_w}.{f_w}}" for m in metrics])
