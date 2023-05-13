"""Wrapper to interact with Docker."""
import os
import subprocess
from datetime import date
from pathlib import Path
from typing import List, Optional

from tools.common.container_volume import ContainerVolume, VolumeMapping
from tools.host.constants import Colors
from tools.host.get_repo_root import get_repo_root

NAME = "dl"
VERSIONS = [
    "2023-01-24",
    "2022-11-28",
    "2022-11-17",
    "2022-04-04",
    "2022-02-05",
]

DOCKERFILE_RELPATH = "tools/conf/Dockerfile"


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


def _docker_run(  # pylint: disable=too-many-arguments,too-many-locals
    cmd: str,
    args: List[str],
    run_from: str,
    volume_mapping: VolumeMapping,
    tf_verbose: bool = True,
    continue_running: bool = True,
    daemon: bool = False,
):
    """Run development docker container."""
    run_params = [
        f"-{'d' if daemon else ''}it",
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
    print(
        f"{Colors.BLUE}Running the command in container "
        f"{latest_image_name_and_tag}{Colors.RESET}"
    )
    print("\t", f"{cmd} {' '.join(args)}")
    subprocess.run(docker_cmd, check=not continue_running)


def docker_run_repo_root(  # pylint: disable=too-many-arguments
    cmd: str,
    args: List[str],
    additional_volumes: Optional[VolumeMapping] = None,
    tf_verbose: bool = True,
    continue_running: bool = True,
    daemon: bool = False,
):
    """Run development docker container from repo root."""
    if additional_volumes is None:
        additional_volumes = {}
    repo_root_on_host = get_repo_root()

    _docker_run(
        cmd,
        args,
        run_from=ContainerVolume.REPO_ROOT,
        volume_mapping={
            repo_root_on_host: ContainerVolume.REPO_ROOT,
            **additional_volumes,
        },
        tf_verbose=tf_verbose,
        continue_running=continue_running,
        daemon=daemon,
    )
