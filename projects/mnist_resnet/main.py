#!/usr/bin/env python3
"""Experiments with mnist and resnet."""
from pathlib import Path

import tensorflow as tf

from core.assembled_model import AssembledModel
from core.experiment import Experiment
from core.settings import DEFAULT_CONFIG_JSON_FILENAME, ExperimentSettings
from core.task.evaluation import Evaluation
from core.task.task_group import TaskGroup
from core.task.training import Training
from projects.mnist_resnet.data.loader import MnistLoader
from projects.mnist_resnet.data.pre_proc import MnistPreProc
from projects.mnist_resnet.objective.mnist_objective import MnistObjective
from zoo.heads.dense import DenseHead
from zoo.selection import make_backbone_from_config

# TODO: add an integration test for train/eval runtime and stats of 1 epoch.


def main():
    """Run an experiment."""
    project_name = Path(__file__).parts[-2]
    settings = ExperimentSettings.load_project_default(project_name)

    settings.serialize(
        f"{settings.output_directory}/{DEFAULT_CONFIG_JSON_FILENAME}", mark_env=True
    )

    objective = MnistObjective()
    optimizer = tf.keras.optimizers.Adam()
    tasks = TaskGroup(
        Training(optimizer, objective),
        Evaluation(objective),
    )

    loader = MnistLoader(
        train_batch_size=settings.config.train_batch_size,
        eval_batch_size=settings.config.eval_batch_size,
        path=settings.data_directory,
        pre_proc=MnistPreProc(),
    )

    model = AssembledModel(
        backbone=make_backbone_from_config(settings.backbone),
        heads={
            MnistLoader.HeadNames.CLASSIFICATION: DenseHead(
                [(1000, "relu"), (10, None)]
            )
        },
        input_signature=loader.make_tensorspec_for_export(),
    )

    experiment = Experiment(
        settings=settings,
        tasks=tasks,
        data_loader=loader,
        model=model,
    )
    experiment.start()


if __name__ == "__main__":
    main()
