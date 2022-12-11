"""Data loader base."""
import abc
from pathlib import Path

import tensorflow as tf

from interfaces.pre_proc import PreProcessing


class Loader(abc.ABC):
    """Data loader base."""

    EXPORT_BATCH_SIZE = 1
    EXPORT_INPUT_NAME = "input"

    def __init__(
        self,
        train_batch_size: int,
        eval_batch_size: int,
        path: Path,
        pre_proc: PreProcessing,
    ):
        """Create an instance."""
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._path = path
        self._pre_proc = pre_proc

        self._train_elems = 0
        self._eval_elems = 0
        self._train_ds, self._eval_ds = self._load()

    @abc.abstractmethod
    def _load(self):
        """Load the dataset."""

    @property
    def training_data_handle(self) -> tf.data.Dataset:
        """Get a handle to training data generator."""
        return self._train_ds

    @property
    def eval_data_handle(self) -> tf.data.Dataset:
        """Get a handle to evaluation data generator."""
        return self._eval_ds

    @property
    def train_num_batches(self) -> int:
        """Get number of training batches."""
        return self._train_elems // self._train_batch_size

    @property
    def eval_num_batches(self) -> int:
        """Get number of evaluation batches."""
        return self._eval_elems // self._eval_batch_size

    def make_tensorspec_for_export(self) -> tf.TensorSpec:
        """Create a tensorspec instance for exports."""
        element_spec = self._eval_ds.element_spec[0]
        return tf.TensorSpec(
            shape=(
                type(self).EXPORT_BATCH_SIZE,  # N
                element_spec.shape[1],  # H
                element_spec.shape[2],  # W
                element_spec.shape[3],  # C
            ),
            dtype=tf.float32,
            name=type(self).EXPORT_INPUT_NAME,
        )
