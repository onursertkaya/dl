"""Wrapper to interact with Docker."""
import os
import subprocess
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from tools.get_repo_root import get_repo_root

NAME = "dl"
VERSIONS = [
    "2022-11-28",
    "2022-11-17",
    "2022-04-04",
    "2022-02-05",
]

REPO_ROOT = "/workdir"
DOCKERFILE_RELPATH = "tools/conf/Dockerfile"

VolumeMapping = Dict[str, str]


def docker_build(tag: Optional[str] = ""):
    """Build development docker image."""
    if not tag:
        tag = date.today().strftime("%Y-%m-%d")
    name_tag = NAME + ":" + tag

    docker_context_dir = Path(get_repo_root())
    dockerfile_dir = docker_context_dir / DOCKERFILE_RELPATH
    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile_dir),
        "-t",
        name_tag,
        str(docker_context_dir),
    ]
    subprocess.run(
        cmd,
        cwd=docker_context_dir,
        check=True,
    )


def _docker_run(
    cmd: str,
    args: List[str],
    run_from: str,
    volume_mapping: VolumeMapping,
    tf_verbose: bool = True,
    continue_running: bool = True,
):
    """Run development docker container."""
    run_params = [
        "-it",
        "--rm",
        "--gpus",
        "all",
    ]

    env_vars = [
        "-e",
        f"PYTHONPATH={run_from}",
        *([] if tf_verbose else ["-e", "TF_CPP_MIN_LOG_LEVEL=3"]),
    ]

    volume_args = zip(
        ["-v"] * len(volume_mapping),
        [f"{host}:{container}" for host, container in volume_mapping.items()],
    )
    cache_volume_args = [
        "-v",
        f"{Path('~').expanduser()}/.cache:/.cache",
    ]

    volumes = [item for sublist in volume_args for item in sublist] + cache_volume_args

    workdir = [
        "-w",
        run_from,
    ]

    user_and_group = [
        "-u",
        f"{os.geteuid()}:{os.getegid()}",
    ]

    latest_image_name_and_tag = f"{NAME}:{VERSIONS[0]}"

    docker_cmd = [
        "docker",
        "run",
        *run_params,
        *user_and_group,
        *env_vars,
        *volumes,
        *workdir,
        latest_image_name_and_tag,
        cmd,
        *args,
    ]
    print(f"Running the command in container {latest_image_name_and_tag}")
    print("\t", f"{cmd} {' '.join(args)}")
    subprocess.run(docker_cmd, check=not continue_running)


def docker_run_repo_root(
    cmd: str,
    args: List[str],
    additional_volumes: Optional[VolumeMapping] = None,
    tf_verbose: bool = True,
    continue_running: bool = True,
):
    """Run development docker container from repo root."""
    if additional_volumes is None:
        additional_volumes = {}
    repo_root_on_host = get_repo_root()

    _docker_run(
        cmd,
        args,
        run_from=REPO_ROOT,
        volume_mapping={
            repo_root_on_host: REPO_ROOT,
            **additional_volumes,
        },
        tf_verbose=tf_verbose,
        continue_running=continue_running,
    )