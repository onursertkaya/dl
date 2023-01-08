#!/usr/bin/env python3
"""Experiments with cifar100 and resnet."""
import tensorflow as tf

from core.assembled_model import AssembledModel
from core.experiment import Experiment
from core.experiment_settings import ExperimentSettings
from core.task.evaluation import Evaluation
from core.task.task_group import TaskGroup
from core.task.training import Training
from projects.cifar100_resnet.data.loader import Cifar100Loader
from projects.cifar100_resnet.data.pre_proc import Cifar100PreProc
from projects.cifar100_resnet.objective.objective import Cifar100Objective
from tools.common.common_project_args import make_argument_parser
from zoo.heads.dense import DenseHead
from zoo.models.resnet import Resnet, ResnetBlockC

# TODO: add an integration test for train/eval runtime of 1 epoch.


def main():
    """Run an experiment."""
    parser = make_argument_parser("mnist")
    args = parser.parse_args()

    settings = ExperimentSettings(
        name=args.name,
        directory=args.experiment_dir,
        epochs=10,
        restore_from=args.restore_from_chkpt,
        train_batch_size=64,
        eval_batch_size=32,
        debug=args.debug,
    )

    objective = Cifar100Objective()
    optimizer = tf.keras.optimizers.Adam()
    tasks = TaskGroup(
        Training(optimizer, objective),
        Evaluation(objective),
    )

    loader = Cifar100Loader(
        train_batch_size=settings.train_batch_size,
        eval_batch_size=settings.eval_batch_size,
        path=args.data_dir,
        pre_proc=Cifar100PreProc(),
    )

    model = AssembledModel(
        backbone=Resnet(34, ResnetBlockC),
        heads={
            Cifar100Loader.HeadNames.CLASSIFICATION: DenseHead(
                [(1000, "relu"), (20, None)]
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
