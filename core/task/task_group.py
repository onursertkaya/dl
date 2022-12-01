"""Group of tasks."""
from dataclasses import dataclass
from typing import Optional

from core.task.evaluation import Evaluation
from core.task.training import Training


@dataclass(frozen=True)
class TaskGroup:
    """Group of tasks."""

    training: Optional[Training] = None
    evaluation: Optional[Evaluation] = None

    def __post_init__(self):
        """Validate the instance.

        All attributes cannot be None at the same time.
        """
        assert any(task is not None for task in [self.training, self.evaluation])
