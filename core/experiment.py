"""Class for orchestrating deep learning experiments."""
from pathlib import Path
from typing import Optional

import tensorflow as tf

from core.export import Export
from core.settings import ExperimentSettings
from core.summary import Summary
from core.task.task import Task
from core.task.task_group import TaskGroup
from interfaces.dataloader import Loader
from zoo.models.base import AssembledModel

# TODO:
# Essential funtionality

# - learning rate scheduling & optimizer management, wrapped.
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
# - ./run.py mount host cache dir
# - replace print statements with logging.info

# Ideas

# - When starting an experiment, commit the changes in a predefined
#   branch (don't work with git.diff files) for traceability.


class Experiment:
    """Experiment manager."""

    def __init__(
        self,
        settings: ExperimentSettings,
        model: AssembledModel,
        tasks: TaskGroup,
        data_loader: Loader,
    ):
        """Initialize an experiment manager."""

        self._settings = settings
        self._model = model
        self._tasks = tasks
        self._loader = data_loader

        self._summary = Summary(self._settings.directory)

        self._exporter: Optional[Export] = None
        if self._tasks.training:
            self._configure_exporter()

        if settings.restore_from:
            import os

            dir_content = set(os.listdir(settings.restore_from))
            assert {"assets", "variables", "saved_model.pb"}.issubset(dir_content)
            print("Restoring entire model with weights and optimizer state...")
            # TODO: ensure loaded model's config (arch) and self._model's are the same.
            #       otherwise, loaded model completely overwrites the model requested by the user.
            #       consider warning the user in this case.
            self._model = tf.keras.models.load_model(settings.restore_from)

        self._register_interrupt_signal_callback()

    def _configure_exporter(self):
        self._exporter = Export(
            onnx_path=self._onnx_path,
            model=self._model,
        )

    def _register_interrupt_signal_callback(self):
        pass
        # signal.signal(signal.SIGINT, self._tasks.training.on_exit)

    def start(self):
        print(f"Starting experiment {self._settings.name}.\n\n\n")

        for epoch_ctr in range(1, 1 + self._settings.epochs):
            self._train_one_epoch(epoch_ctr)
            self._evaluate_one_epoch(epoch_ctr)
            self._perform_post_epoch_ops(epoch_ctr)
        print("Experiment finished. Exporting...")
        if self._exporter:
            self._exporter.create_onnx()
        print("Done")

    def _train_one_epoch(self, epoch_ctr: int):
        batch_ctr = 1
        if self._tasks.training:
            for batch in self._loader.training_data_handle:
                data_points, named_labels = batch
                self._tasks.training.step(self._model, data_points, named_labels)
                self._print(
                    self._tasks.training,
                    batch_ctr,
                    self._loader.train_num_batches,
                    epoch_ctr,
                )
                self._log_summary_to_disk(
                    self._tasks.training, batch_ctr, self._loader.train_num_batches
                )
                batch_ctr += 1

    def _evaluate_one_epoch(self, epoch_ctr: int):
        batch_ctr = 1
        should_evaluate = epoch_ctr % self._settings.train_eval_freq_ratio == 0
        if self._tasks.evaluation and should_evaluate:
            for batch in self._loader.eval_data_handle:
                data_points, named_labels = batch
                self._tasks.evaluation.step(self._model, data_points, named_labels)
                self._print(
                    self._tasks.evaluation,
                    batch_ctr,
                    self._loader.eval_num_batches,
                    epoch_ctr,
                )
                self._log_summary_to_disk(
                    self._tasks.evaluation, batch_ctr, self._loader.eval_num_batches
                )
                batch_ctr += 1

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

    def _print(
        self, task: Task, batch_ctr: int, num_total_task_batches: int, epoch_ctr
    ):
        is_modulo = self._check_modulo(
            batch_ctr, num_total_task_batches, self._settings.stats_prints_per_epoch
        )
        is_first_or_last_batch = batch_ctr == num_total_task_batches or batch_ctr == 1
        if is_modulo or is_first_or_last_batch:
            task.print_stats(
                epoch_ctr,
                self._settings.epochs,
                batch_ctr,
                num_total_task_batches,
            )

    def _log_summary_to_disk(
        self, task: Task, batch_ctr: int, num_total_task_batches: int
    ):
        if self._check_modulo(
            batch_ctr,
            num_total_task_batches,
            self._settings.stats_loggings_per_epoch,
        ):
            self._summary.write(task)

    @staticmethod
    def _check_modulo(
        batch_ctr: int, num_total_task_batches: int, frequency: int
    ) -> bool:
        return batch_ctr % (num_total_task_batches // frequency) == 0

    @property
    def _onnx_path(self):
        return (
            Path(self._settings.directory) / "onnx_export" / self._settings.name
        ).with_suffix(".onnx")

    @property
    def _checkpoint_path(self) -> Path:
        return Path(self._settings.directory) / "checkpoints"
