from typing import final

import tensorflow as tf

from core.task.task import Task


@final
class Inference(Task):
    """Inference mode."""

    def __init__(self):
        """Initialize inference mode."""

    def step(self, data):
        images, _ = data
        predictions = self._model(images, training=False)
        # save predictions to disk? numpy, h5py
        # visualize, somehow?

    def print_stats(self, epoch: int, epochs: int, batch: int, batches: int):
        pass
