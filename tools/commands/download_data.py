"""Utility class to download various datasets."""
from pathlib import Path

from tools.commands.docker_commands import docker_run_repo_root
from tools.get_repo_root import get_repo_root


def download(dataset: str, target_path: str):
    """Download the dataset to target path.

    This function is intended to be called outside of the development
    container. It then forwards this call to the container and calls
    the same file as a module, where the wrapped function is called.
    """
    file_relpath = str(Path(__file__).relative_to(get_repo_root()))
    docker_run_repo_root(
        "python3", [file_relpath, dataset, target_path], {target_path: target_path}
    )


def _download_in_docker(dataset: str, target_path: str):
    # pylint: disable=import-outside-toplevel
    import tensorflow as tf

    supported_dataset_loaders = {"mnist": tf.keras.datasets.mnist.load_data}
    assert dataset in supported_dataset_loaders
    loader = supported_dataset_loaders[dataset]
    loader(path=f"{target_path}/mnist.npz")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("target_path")
    args = parser.parse_args()
    _download_in_docker(args.dataset_name, args.target_path)
