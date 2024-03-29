"""Mnist data loader."""
from enum import Enum
from pathlib import Path
from typing import final

import tensorflow as tf

from interfaces.dataloader import Loader
from interfaces.pre_proc import PreProcessing


@final
class MnistLoader(Loader):
    """Mnist loader."""

    class HeadNames(str, Enum):
        """Head names."""

        CLASSIFICATION = "digit_category"

    def __init__(
        self,
        path: Path,
        train_batch_size: int,
        eval_batch_size: int,
        pre_proc: PreProcessing,
    ):
        """Create an instance."""
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            path=path,
            pre_proc=pre_proc,
        )

    def _load(self):
        (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data(
            path=f"{self._path}/mnist.npz"
        )

        self._train_elems = len(x_train)
        self._eval_elems = len(x_eval)

        x_train_processed, x_eval_processed = self._pre_proc(x_train, x_eval)

        train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (x_train_processed, {self.HeadNames.CLASSIFICATION: y_train})
            )
            .shuffle(10000)
            .batch(self._train_batch_size)
        )

        eval_ds = tf.data.Dataset.from_tensor_slices(
            (x_eval_processed, {self.HeadNames.CLASSIFICATION: y_eval})
        ).batch(self._eval_batch_size)

        return train_ds, eval_ds
