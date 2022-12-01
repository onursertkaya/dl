#!/usr/bin/env python3
"""Experiments with mnist and resnet."""
import argparse

import tensorflow as tf

from core.experiment import Experiment
from core.experiment_settings import ExperimentSettings
from core.task.evaluation import Evaluation
from core.task.task_group import TaskGroup
from core.task.training import Training
from projects.mnist_resnet.data.loader import MnistLoader
from projects.mnist_resnet.objective.head import MnistDenseHead
from projects.mnist_resnet.objective.mnist_objective import MnistObjective
from zoo.models.base import AssembledModel
from zoo.models.resnet import Resnet, ResnetBlockC


def _parse_args():
    parser = argparse.ArgumentParser("Start a mnist experiment.")

    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-t", "--tasks", type=str)
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-e", "--experiment-dir", type=str)
    parser.add_argument("-r", "--restore-from-chkpt", type=str, default=None)

    return parser.parse_args()


def main():
    """Run an experiment."""
    args = _parse_args()

    settings = ExperimentSettings(
        name=args.name,
        directory=args.experiment_dir,
        epochs=20,
        restore_from=args.restore_from_chkpt,
        train_batch_size=64,
        eval_batch_size=32,
    )

    objective = MnistObjective()
    optimizer = tf.keras.optimizers.Adam()
    tasks = TaskGroup(
        Training(optimizer, objective),
        Evaluation(objective),
    )

    loader = MnistLoader(
        train_batch_size=settings.train_batch_size,
        eval_batch_size=settings.eval_batch_size,
        path=args.data_dir,
    )

    model = AssembledModel(
        backbone=Resnet(34, ResnetBlockC),
        heads={MnistLoader.HeadNames.CLASSIFICATION: MnistDenseHead()},
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
