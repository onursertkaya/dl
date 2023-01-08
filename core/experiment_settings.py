"""Experiment settings."""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class ExperimentSettings:  # pylint: disable=too-many-instance-attributes
    """Experiment settings."""

    name: str
    directory: str

    epochs: int

    train_batch_size: int
    eval_batch_size: int

    train_eval_freq_ratio: int = 1
    model_serialization_freq: int = 1

    prints_per_epoch: int = 10
    disk_writes_per_epoch: int = 10

    date_time: str = datetime.now().isoformat()
    unique_identifier: str = str(uuid4())

    debug: bool = False
    restore_from: Optional[str] = None
