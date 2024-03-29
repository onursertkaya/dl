"""Module for mapping host to container environment."""
from typing import Dict

VolumeMapping = Dict[str, str]


class ContainerVolume:
    """A container volume."""

    DATA_DIR = "/data"
    EXPERIMENT_DIR = "/experiment"
    CHECKPOINT_DIR = "/checkpoint"
    REPO_ROOT = "/workdir"

    def __init__(self, output_dir: str, data_dir: str, checkpoint_dir: str):
        """Initialize."""
        self._output = output_dir
        self._data = data_dir
        self._checkpoint = checkpoint_dir

    def make_mapping(self) -> VolumeMapping:
        """Create a host-container path mapping."""
        return {
            self._data: ContainerVolume.DATA_DIR,
            self._output: ContainerVolume.EXPERIMENT_DIR,
            **(
                {self._checkpoint: ContainerVolume.CHECKPOINT_DIR}
                if self._checkpoint
                else {}
            ),
        }
