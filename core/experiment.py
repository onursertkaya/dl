"""Class for orchestrating deep learning experiments."""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import tensorflow as tf

from core.assembled_model import AssembledModel
from core.experiment_settings import ExperimentSettings
from core.export import Export
from core.summary import Summary
from core.task.evaluation import Evaluation
from core.task.task import Task
from core.task.task_group import TaskGroup
from core.task.training import Training
from interfaces.dataloader import Loader

# TODO:
# Essential funtionality

# - add model loading -> inference support
#  - should not be a Task, as there would be no functionality reuse.
# - learning rate scheduling & optimizer management, wrapped in TrainingPolicy class.
# - augmentation, generalized within Preprocessing class
# - base class for HyperParamSettings, each project needs to override
#   - yaml serializer/deserializer for settings classes
# - define scenarios for model loading as the implementation is going to differ
#   - modifying the loaded model
#       - finetuning
#         - with same arch
#         - with modified arch towards the end (i.e. just restore the backbone)
#       - restore interrupted training:
#         - model + optimizer + dataset iterator needs to be reloaded

#   - read-only scenarios
#     - eval/infer/use network output for training other network (same as infer):
#       - just the model
#   - more??

# Chores

# - fixed seeds, np and tf
# - replace print statements with logging.info
# - resolve the absl warning for traced functions

# Ideas

# - When starting an experiment, commit the changes in a predefined
#   branch (don't work with git.diff files) for traceability.


class Experiment:
    """A deep learning experiment."""

    class InvalidExperimentTask(Exception):
        """Raise when Taskgroup contains an invalid task type."""

    def __init__(
        self,
        settings: ExperimentSettings,
        tasks: TaskGroup,
        data_loader: Loader,
        model: AssembledModel,
    ):
        """Initialize an experiment."""
        self._settings = settings
        self._tasks = tasks
        self._loader = data_loader

        self._model = self._restore_model() if self._settings.restore_from else model
        self._exporter = (
            Export(
                onnx_path=self._onnx_path,
                model=self._model,
            )
            if self._tasks.training
            else None
        )

        self._summary = Summary(self._settings, self._tasks)
        self._register_interrupt_signal_callback()

    def start(self):
        """Start an experiment."""
        print(
            f"Starting experiment {self._settings.name} {datetime.now().isoformat()}.\n\n\n"
        )

        for epoch_ctr in range(1, 1 + self._settings.epochs):
            self._train_one_epoch(epoch_ctr)
            self._evaluate_one_epoch(epoch_ctr)
            self._perform_post_epoch_ops(epoch_ctr)
        print("Experiment finished.")
        if self._exporter:
            print("Exporting...")
            self._exporter.create_onnx()
            print("Done")

    def _train_one_epoch(self, epoch_ctr: int):
        self._perform_task(epoch_ctr, self._tasks.training)

    def _evaluate_one_epoch(self, epoch_ctr: int):
        self._perform_task(epoch_ctr, self._tasks.evaluation)

    def _perform_task(self, epoch_ctr: int, task: Optional[Task]):
        batch_ctr = 1

        data_handle: tf.data.Dataset
        num_batches: int
        should_perform: bool
        if task is None:
            return

        if isinstance(task, Training):
            data_handle = self._loader.training_data_handle
            num_batches = self._loader.train_num_batches
            should_perform = self._tasks.training is not None
        elif isinstance(task, Evaluation):
            data_handle = self._loader.eval_data_handle
            num_batches = self._loader.eval_num_batches
            should_perform = (self._tasks.evaluation is not None) and (
                epoch_ctr % self._settings.train_eval_freq_ratio == 0
            )
        else:
            raise Experiment.InvalidExperimentTask(type(task))

        if should_perform:
            for batch in data_handle:
                data_points, named_labels = batch
                task.step(self._model, data_points, named_labels)
                self._summary.write(
                    task,
                    num_batches,
                    batch_ctr,
                    epoch_ctr,
                )
                batch_ctr += 1

    def _restore_model(self):
        dir_content = set(os.listdir(self._settings.restore_from))
        assert {"assets", "variables", "saved_model.pb"}.issubset(dir_content)
        print("Restoring entire model with weights and optimizer state...")
        # TODO: ensure loaded model's config (arch) and self._model's are the same.
        #       otherwise, loaded model completely overwrites the model requested by the user.
        #       consider warning the user in this case.
        self._model = tf.keras.models.load_model(self._settings.restore_from)

    def _register_interrupt_signal_callback(self):
        pass
        # TODO: add signal.signal(signal.SIGINT, self._tasks.training.on_exit)

    def _perform_post_epoch_ops(self, epoch_ctr: int):
        if self._tasks.training:
            self._tasks.training.reset_states()

            should_save_checkpoint = (
                epoch_ctr % self._settings.model_serialization_freq == 0
            )
            if should_save_checkpoint:
                checkpoint_dir = self._checkpoint_path / f"e-{epoch_ctr:0>3}"
                print(f"Saving checkpoint to {checkpoint_dir}")
                # self._model.save_weights(self._checkpoint_path)
                self._model.save(checkpoint_dir)

    @property
    def _onnx_path(self):
        return (
            Path(self._settings.directory) / "onnx_export" / self._settings.name
        ).with_suffix(".onnx")

    @property
    def _checkpoint_path(self) -> Path:
        return Path(self._settings.directory) / "checkpoints"
