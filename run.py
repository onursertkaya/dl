#!/usr/bin/env python3
"""Run an experiment inside docker container."""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))  # noqa: E402
# pylint: disable=wrong-import-position
from tools.commands.check import run_py_checks
from tools.commands.docker_commands import docker_build, docker_run_repo_root
from tools.commands.experiment import PROJECTS_DIR_RELPATH, start_experiment
from tools.commands.tests import run_py_tests

# pylint: enable=wrong-import-position


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
        [project.name for project in Path(PROJECTS_DIR_RELPATH).iterdir()]
    )
    parser_experiment.add_argument(
        "-p", "--project", type=str, required=True, choices=current_projects
    )
    parser_experiment.add_argument(
        "--project-args", nargs=argparse.REMAINDER, default=[]
    )

    # Other commands
    subparsers.add_parser("check", help="Run checks and formatting.")

    parser_download = subparsers.add_parser("download", help="Download a dataset.")
    parser_download.add_argument("name_and_path", nargs="+")

    subparsers.add_parser("tests", help="Run tests.")
    subparsers.add_parser("docker_build", help="Build the docker image.")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return parser.parse_args()


if __name__ == "__main__":
    start = datetime.datetime.now()

    args = _parse_args()

    if args.subparser == "experiment":
        start_experiment(args)
    elif args.subparser == "check":
        run_py_checks()
    elif args.subparser == "download":
        docker_run_repo_root(
            cmd="python3",
            args=args.name_and_path,
            additional_volumes={args.name_and_path[-1]: args.name_and_path[-1]},
        )
    elif args.subparser == "tests":
        run_py_tests()
    elif args.subparser == "docker_build":
        docker_build()
    else:
        raise RuntimeError(f"Invalid argument to {__file__}")

    end = datetime.datetime.now()
    print(f"{__file__}: Total runtime:", (end - start))
