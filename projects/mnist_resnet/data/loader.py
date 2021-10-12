from enum import Enum
from pathlib import Path
from typing import final

import tensorflow as tf

from data_types.spatial_size import SpatialSize
from interfaces.dataloader import Loader


@final
class MnistLoader(Loader):
    """Mnist loader."""

    class HeadNames(Enum, str):
        classification = "digit_class"

    def __init__(self, train_batch_size: int, eval_batch_size: int, path: Path):
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            path=path,
        )

        (x_train, y_train), (x_eval, y_eval) = tf.keras.datasets.mnist.load_data(
            path=f"{self._path}/mnist.npz"
        )
        x_train, x_eval = x_train / 255.0, x_eval / 255.0

        x_train = x_train[..., tf.newaxis].astype("float32")
        x_eval = x_eval[..., tf.newaxis].astype("float32")

        self._train_elems = len(x_train)
        self._eval_elems = len(x_eval)

        self._train_spatial_size = SpatialSize(
            width=x_train.shape[2], height=x_train.shape[1]
        )
        self._eval_spatial_size = SpatialSize(
            width=x_eval.shape[2], height=x_eval.shape[1]
        )

        self._train_ds = (
            tf.data.Dataset.from_tensor_slices(
                (x_train, {self.HeadNames.classification: y_train})
            )
            .shuffle(10000)
            .batch(train_batch_size)
        )

        self._eval_ds = tf.data.Dataset.from_tensor_slices(
            (x_eval, {self.HeadNames.classification: y_eval})
        ).batch(eval_batch_size)

    @property
    def training_data_handle(self) -> tf.data.Dataset:
        return self._train_ds

    @property
    def eval_data_handle(self) -> tf.data.Dataset:
        return self._eval_ds

    @property
    def train_num_batches(self) -> int:
        return self._train_elems // self._train_batch_size

    @property
    def eval_num_batches(self) -> int:
        return self._eval_elems // self._eval_batch_size

    @property
    def train_spatial_size(self) -> SpatialSize:
        return self._train_spatial_size

    @property
    def eval_spatial_size(self) -> SpatialSize:
        return self._eval_spatial_size