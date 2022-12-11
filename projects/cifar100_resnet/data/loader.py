"""Cifar100 data loader."""
import pickle
from enum import Enum
from pathlib import Path
from typing import Tuple, final

import numpy as np
import tensorflow as tf

from interfaces.dataloader import Loader
from interfaces.pre_proc import PreProcessing


@final
class Cifar100Loader(Loader):
    """Cifar100 loader."""

    class HeadNames(str, Enum):
        """Head names."""

        CLASSIFICATION = "object_category"

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
        x_train, y_train = self._load_as_channels_last("train")
        x_eval, y_eval = self._load_as_channels_last("test")

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

    def _load_as_channels_last(self, pickle_name: str) -> Tuple[np.ndarray, np.ndarray]:
        images, labels = Cifar100Loader._unpickle(
            f"{self._path}/cifar-100-python/{pickle_name}"
        )
        # linear row-major -> spatial row-major (i.e. nchw) -> nhwc
        return (
            images.reshape((len(images), 3, 32, 32)).swapaxes(1, 3).swapaxes(1, 2),
            labels,
        )

    @staticmethod
    def _unpickle(pickle_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # copied from https://www.cs.toronto.edu/~kriz/cifar.html, slightly changed
        with open(pickle_path, "rb") as pickle_file:
            data_dict = pickle.load(pickle_file, encoding="bytes")
        return np.array(data_dict[b"data"], dtype=np.float32), np.array(
            data_dict[b"coarse_labels"], dtype=np.uint8
        )
