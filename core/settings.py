"""Various settings definitions."""
from dataclasses import asdict, dataclass
from datetime import datetime
from json import dump, load
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

DEFAULT_CONFIG_JSON_FILENAME = "default_config.json"


@dataclass(frozen=True)
class Config:
    """Experiment configuration."""

    epochs: int = 10

    train_batch_size: int = 32
    eval_batch_size: int = 32

    train_eval_freq_ratio: int = 1
    model_serialization_freq: int = 1

    prints_per_epoch: int = 10
    disk_writes_per_epoch: int = 10

    debug: bool = False


@dataclass(frozen=True)
class BackboneConfig:
    """Model backbone configuration."""

    name: str
    init_kwargs: Dict[str, Any]


@dataclass(frozen=True)
class ExperimentSettings:  # pylint: disable=too-many-instance-attributes
    """Experiment settings."""

    output_directory: str
    data_directory: str
    backbone: BackboneConfig

    restore_from: Optional[str] = None
    config: Config = Config()

    def serialize(self, path: str, mark_env: bool):
        """Write to disk."""
        assert path.endswith(".json"), f"{path} should have .json suffix."
        mark = (
            {"date_time": datetime.now().isoformat(), "uid": str(uuid4())}
            if mark_env
            else {}
        )
        with open(path, "w", encoding="utf-8") as json_file:
            dump({**asdict(self), **mark}, fp=json_file, indent=4)

    @classmethod
    def load_project_default(cls, project: str):
        """Load the default configuration file of project."""
        return cls.load(f"projects/{project}/{DEFAULT_CONFIG_JSON_FILENAME}")

    @classmethod
    def load(cls, path: str):
        """Load from disk."""
        assert path.endswith(".json"), f"{path} should have .json suffix."
        with open(path, "r", encoding="utf-8") as json_file:
            loaded_dict = load(json_file)
            config = Config(**loaded_dict["config"])
            backbone = BackboneConfig(**loaded_dict["backbone"])
            del loaded_dict["config"]
            del loaded_dict["backbone"]
            return cls(
                **loaded_dict,
                backbone=backbone,
                config=config,
            )

    def validate(self):
        """Validate the fields.

        Should only be invoked on instances that are to be used in experiments.
        """
        assert Path(self.output_directory).is_dir()
        assert Path(self.data_directory).is_dir()
        if self.restore_from is not None:
            assert Path(self.restore_from).is_dir()
