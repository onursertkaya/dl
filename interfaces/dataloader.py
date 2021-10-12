import abc
from pathlib import Path

import tensorflow as tf

from data_types.spatial_size import SpatialSize


class Loader(abc.ABC):
    """Base class for data loading."""

    def __init__(self, train_batch_size: int, eval_batch_size: int, path: Path):

        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._path = path

    @abc.abstractproperty
    def training_data_handle(self) -> tf.data.Dataset:
        pass

    @abc.abstractproperty
    def eval_data_handle(self) -> tf.data.Dataset:
        pass

    @abc.abstractproperty
    def train_num_batches(self) -> int:
        pass

    @abc.abstractproperty
    def eval_num_batches(self) -> int:
        pass

    @abc.abstractproperty
    def train_spatial_size(self) -> SpatialSize:
        pass

    @abc.abstractproperty
    def eval_spatial_size(self) -> SpatialSize:
        pass
