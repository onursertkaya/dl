"""Data loader base."""
import abc
from pathlib import Path

import tensorflow as tf

from data_types.spatial_size import SpatialSize


class Loader(abc.ABC):
    """Data loader base."""

    EXPORT_BATCH_SIZE = 1

    def __init__(self, train_batch_size: int, eval_batch_size: int, path: Path):
        """Create an instance."""
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._path = path

    @property
    @abc.abstractmethod
    def training_data_handle(self) -> tf.data.Dataset:
        """Get a handle to training data generator."""

    @property
    @abc.abstractmethod
    def eval_data_handle(self) -> tf.data.Dataset:
        """Get a handle to evaluation data generator."""

    @property
    @abc.abstractmethod
    def train_num_batches(self) -> int:
        """Get number of training batches."""

    @property
    @abc.abstractmethod
    def eval_num_batches(self) -> int:
        """Get number of evaluation batches."""

    @property
    @abc.abstractmethod
    def train_spatial_size(self) -> SpatialSize:
        """Get the spatial size of the training input."""

    @property
    @abc.abstractmethod
    def eval_spatial_size(self) -> SpatialSize:
        """Get the spatial size of the evaluation input."""

    def make_tensorspec_for_export(self) -> tf.TensorSpec:
        """Create a tensorspec instance for exports."""
        return tf.TensorSpec(
            shape=(
                type(self).EXPORT_BATCH_SIZE,
                self.eval_spatial_size.height,
                self.eval_spatial_size.width,
                self.eval_spatial_size.depth,
            ),
            dtype=tf.float32,
            name="input",
        )
