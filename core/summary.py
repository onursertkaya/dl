from typing import Iterable

import tensorflow as tf

from core.task.task import Task

MetricsList = Iterable[tf.keras.metrics.Metric]


class Summary:
    """Tensorboard summary utilities."""

    def __init__(self, logdir: str):
        self._write_ctr = 0
        self._writer = tf.summary.create_file_writer(logdir + "/" + "tboard")

    def write(self, task: Task):
        with self._writer.as_default():
            with tf.name_scope(task.__class__.__name__):
                for loss in task.loss_metrics:
                    tf.summary.scalar(loss.name, loss.result(), step=self._write_ctr)
                for pm in task.performance_metrics:
                    tf.summary.scalar(pm.name, pm.result(), step=self._write_ctr)
                self._write_ctr += 1
