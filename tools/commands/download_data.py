"""Utility class to download various datasets."""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests  # type: ignore
import tensorflow as tf


@dataclass(frozen=True)
class Dataset:
    """A dataset."""

    name: str
    file_extension: str
    downloader: Callable
    url: Optional[str] = None

    def download(self, target_dir: str):
        """Download dataset to target path."""
        path = Path(target_dir) / f"{self.name}.{self.file_extension}"
        if self.url:
            self.downloader(url=self.url, path=path)
        else:
            self.downloader(path=path)


def _download_file(url: str, path: str):
    """Download file."""
    target = Path(path)
    assert not target.exists()
    req = requests.get(url)
    with open(target, "wb") as file:
        file.write(req.content)
    assert Path(target).is_file()


def _download(dataset: str, target_dir: str):
    supported_datasets = (
        Dataset(
            "mnist",
            "npz",
            tf.keras.datasets.mnist.load_data,
        ),
        Dataset(
            "cifar100",
            "tar.gz",
            _download_file,
            "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        ),
    )
    picked = list(filter(lambda dset: dset.name == dataset, supported_datasets))
    assert len(picked) == 1
    picked[0].download(target_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("target_dir")
    args = parser.parse_args()
    _download(args.dataset_name, args.target_dir)
