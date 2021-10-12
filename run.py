#!/usr/bin/env python3
"""Run an experiment inside docker container."""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))
from tools.docker.commands import (
    CHECKPOINT_DIR,
    DATA_DIR,
    EXPERIMENT_DIR,
    docker_run_repo_root,
)

PROJECTS_RELPATH = "projects"


def _parse_args():
    parser = argparse.ArgumentParser("Run a job.")
    subparsers = parser.add_subparsers(help="", dest="subparser")
    parser_experiment = subparsers.add_parser("experiment", help="Start an experiment.")

    # Experiment settings, required.
    parser_experiment.add_argument("-n", "--name", type=str, required=True)
    parser_experiment.add_argument("-t", "--tasks", type=str, required=True)
    parser_experiment.add_argument("-d", "--data-dir", type=str, required=True)
    parser_experiment.add_argument("-e", "--experiment-dir", type=str, required=True)
    parser_experiment.add_argument(
        "-r", "--restore-from-chkpt", type=str, default=None, required=False
    )

    # Experiment settings, optional.
    parser_experiment.add_argument(
        "-v", "--verbose-tf", action="store_true", required=False
    )

    # Project settings
    current_projects = sorted(
        [project.name for project in Path(PROJECTS_RELPATH).iterdir()]
    )
    parser_experiment.add_argument(
        "-p", "--project", type=str, required=True, choices=current_projects
    )
    parser_experiment.add_argument("--project-args", nargs=argparse.REMAINDER)

    # Other commands
    subparsers.add_parser("check", help="Run checks and formatting.")
    subparsers.add_parser("tests", help="Run tests.")
    subparsers.add_parser("docker_build", help="Build the docker image.")

    args = parser.parse_args()

    return args


def _start_experiment(args):
    experiment_name = args.name
    experiment_tasks = args.tasks
    experiment_dir = args.experiment_dir
    data_dir = args.data_dir
    checkpoint_to_restore = args.restore_from_chkpt
    project_args = args.project_args

    assert Path(data_dir).is_dir()
    Path(experiment_dir).mkdir(exist_ok=True, parents=True)
    assert any([task in args.tasks.split(",") for task in ["train", "eval", "infer"]])
    if checkpoint_to_restore:
        assert Path(checkpoint_to_restore)

    docker_run_repo_root(
        cmd="python3",
        args=[
            f"{PROJECTS_RELPATH}/{args.project}/main.py",
            "-n",
            experiment_name,
            "-t",
            experiment_tasks,
            "-d",
            DATA_DIR,
            "-e",
            EXPERIMENT_DIR,
            *(["-r", CHECKPOINT_DIR] if checkpoint_to_restore else []),
            *project_args,
        ],
        additional_volumes={
            data_dir: DATA_DIR,
            experiment_dir: EXPERIMENT_DIR,
            **(
                {checkpoint_to_restore: CHECKPOINT_DIR} if checkpoint_to_restore else {}
            ),
        },
        tf_verbose=args.verbose_tf,
    )


def _run_checks():
    from tools.py_checks.check import run_py_checks

    run_py_checks()


def _run_tests():
    from tools.py_checks.check import run_py_tests

    run_py_tests()


def _build_docker_image():
    from tools.docker.commands import docker_build

    docker_build()


if __name__ == "__main__":
    start = datetime.datetime.now()

    args = _parse_args()

    if args.subparser == "experiment":
        _start_experiment(args)
    elif args.subparser == "check":
        _run_checks()
    elif args.subparser == "tests":
        _run_tests()
    elif args.subparser == "docker_build":
        _build_docker_image()
    else:
        raise RuntimeError(f"Invalid argument to {__file__}")

    end = datetime.datetime.now()
    print("Total runtime:", (end - start))
