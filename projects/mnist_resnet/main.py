#!/usr/bin/env python3
import argparse

import tensorflow as tf

from core.experiment import Experiment
from core.settings import ExperimentSettings
from core.task.evaluation import Evaluation
from core.task.task_group import TaskGroup
from core.task.training import Training
from projects.mnist_resnet.data.loader import MnistLoader
from projects.mnist_resnet.objective.head import MnistDenseHead
from projects.mnist_resnet.objective.mnist_metrics import MnistLoss
from zoo.models.base import AssembledModel
from zoo.models.resnet import (
    BottleneckBlock,
    Resnet,
    ResnetBlockA,
    ResnetBlockB,
    ResnetBlockC,
)


def _parse_args():
    parser = argparse.ArgumentParser("Start a mnist experiment.")

    parser.add_argument("-n", "--name", type=str)
    parser.add_argument("-t", "--tasks", type=str)
    parser.add_argument("-d", "--data-dir", type=str)
    parser.add_argument("-e", "--experiment-dir", type=str)
    parser.add_argument("-r", "--restore-from-chkpt", type=str, default=None)
    parser.add_argument(
        "--residual-block",
        required=True,
        choices=[
            block.__name__
            for block in [ResnetBlockA, ResnetBlockB, ResnetBlockC, BottleneckBlock]
        ],
    )
    parser.add_argument(
        "--depth",
        required=True,
        type=int,
        choices=[int(key) for key in Resnet.CONFIGS.keys()],
    )

    args = parser.parse_args()
    args.residual_block = globals()[args.residual_block]
    return args


def main():
    args = _parse_args()

    # todo: push to args
    batch_size = 32
    export_batch_size = 1

    settings = ExperimentSettings(
        name=args.name,
        directory=args.experiment_dir,
        epochs=2,
        restore_from=args.restore_from_chkpt,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )

    loss = MnistLoss()
    optimizer = tf.keras.optimizers.Adam()
    tasks = TaskGroup(
        Training(optimizer, loss),
        Evaluation(loss),
    )

    loader = MnistLoader(
        train_batch_size=settings.train_batch_size,
        eval_batch_size=settings.eval_batch_size,
        path=args.data_dir,
    )

    model = AssembledModel(
        backbone=Resnet(args.depth, args.residual_block),
        heads={MnistLoader.HeadNames.classification: MnistDenseHead()},
        input_signature=tf.TensorSpec(
            shape=(
                export_batch_size,
                loader.eval_spatial_size.height,
                loader.eval_spatial_size.width,
                loader.eval_spatial_size.depth,
            ),
            dtype=tf.float32,
            name="input",
        ),
    )

    experiment = Experiment(
        settings=settings,
        model=model,
        tasks=tasks,
        data_loader=loader,
    )
    experiment.start()


if __name__ == "__main__":
    main()
