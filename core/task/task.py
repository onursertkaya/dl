import abc
from typing import Tuple

import tensorflow as tf


class Task(abc.ABC):
    """Base class for tasks."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def print_stats(self, epoch: int, epochs: int, batch: int, batches: int):
        pass

    @abc.abstractproperty
    def loss_metrics(self) -> Tuple[tf.keras.metrics.Metric, ...]:
        pass

    @abc.abstractproperty
    def performance_metrics(self) -> Tuple[tf.keras.metrics.Metric, ...]:
        pass
