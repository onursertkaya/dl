#!/usr/bin/env python3
"""Run an experiment inside docker container."""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.resolve()))  # noqa: E402
# pylint: disable=wrong-import-position
from tools.host.commands.check import run_py_checks
from tools.host.commands.docker_commands import docker_build, docker_run_repo_root
from tools.host.commands.experiment import PROJECTS_DIR_RELPATH, start_experiment
from tools.host.commands.test import run_py_tests

# pylint: enable=wrong-import-position


class ArgModes:
    """Constants for argument run modes."""

    EXPERIMENT = "experiment"
    CHECK = "check"
    GENERATE_DEFAULT_CONFIGS = "generate_default_configs"
    DOWNLOAD = "download"
    TEST = "test"
    DOCKER_BUILD = "docker_build"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"


def _parse_args():
    parser = argparse.ArgumentParser("Run a job.")

    subparsers = parser.add_subparsers(help="", dest="subparser")
    parser_experiment = subparsers.add_parser(
        ArgModes.EXPERIMENT, help="Start an experiment."
    )

    parser_experiment.add_argument(
        "-v", "--verbose-tf", action="store_true", required=False
    )
    parser_experiment.add_argument(
        "-p",
        "--project",
        type=str,
        required=True,
        choices=[project.name for project in Path(PROJECTS_DIR_RELPATH).iterdir()],
    )

    # Other commands
    subparsers.add_parser(ArgModes.CHECK, help="Run checks and formatting.")

    parser_download = subparsers.add_parser(
        ArgModes.DOWNLOAD, help="Download a dataset."
    )
    parser_download.add_argument("name_and_path", nargs="+")

    subparsers.add_parser(ArgModes.TEST, help="Run tests.")
    subparsers.add_parser(ArgModes.DOCKER_BUILD, help="Build the docker image.")
    subparsers.add_parser(
        ArgModes.BACKGROUND, help="Run the docker image in the background."
    )
    subparsers.add_parser(
        ArgModes.INTERACTIVE, help="Run the docker image interactively."
    )
    subparsers.add_parser(ArgModes.GENERATE_DEFAULT_CONFIGS, help="Generate configs.")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return parser.parse_args()


if __name__ == "__main__":
    start = datetime.datetime.now()

    args = _parse_args()

    if args.subparser == ArgModes.EXPERIMENT:
        start_experiment(args.project, args.verbose_tf)
    elif args.subparser == ArgModes.CHECK:
        run_py_checks()
    elif args.subparser == ArgModes.GENERATE_DEFAULT_CONFIGS:
        docker_run_repo_root(
            cmd="python3",
            args=["tools/container/generate_project_default_configs.py"],
        )
    elif args.subparser == ArgModes.DOWNLOAD:
        Path(args.name_and_path[1]).mkdir(parents=True)
        docker_run_repo_root(
            cmd="python3",
            args=["tools/container/download_data.py", *args.name_and_path],
            additional_volumes={args.name_and_path[-1]: args.name_and_path[-1]},
        )
    elif args.subparser == ArgModes.TEST:
        run_py_tests()
    elif args.subparser == ArgModes.DOCKER_BUILD:
        docker_build()
    elif args.subparser == ArgModes.INTERACTIVE:
        docker_run_repo_root(
            cmd="/bin/bash",
            args=[],
        )
    elif args.subparser == ArgModes.BACKGROUND:
        docker_run_repo_root(
            cmd="/bin/bash",
            args=[],
            daemon=True,
        )
    else:
        raise RuntimeError(f"Invalid argument to {__file__}")

    end = datetime.datetime.now()
    print(f"{__file__}: Total runtime:", (end - start))
