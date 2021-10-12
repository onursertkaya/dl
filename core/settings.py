from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class ExperimentSettings:

    name: str
    directory: str

    epochs: int

    train_batch_size: int
    eval_batch_size: int

    train_eval_freq_ratio: int = 1
    model_serialization_freq: int = 1

    stats_prints_per_epoch: int = 10
    stats_loggings_per_epoch: int = 10

    date_time: str = datetime.now().isoformat()
    unique_identifier: str = str(uuid4())

    restore_from: Optional[str] = None
