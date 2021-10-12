"""Wrapper to interact with Docker."""
import os
import subprocess
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from tools.get_repo_root import get_repo_root

NAME = "dl-practice"
VERSIONS = [
    "2022-04-04",
    "2022-02-05",
]

REPO_ROOT = "/workdir"
DATA_DIR = "/data"
EXPERIMENT_DIR = "/experiment"
CHECKPOINT_DIR = "/checkpoint"


def docker_build(tag: Optional[str] = ""):
    """Build development docker image."""
    if not tag:
        tag = date.today().strftime("%Y-%m-%d")
    name_tag = NAME + ":" + tag
    docker_cmd = ["docker", "build", "-t", name_tag, "."]
    dockerfile_dir = Path(__file__).parent
    assert (dockerfile_dir / "Dockerfile").is_file()
    return subprocess.run(docker_cmd, cwd=dockerfile_dir)


def docker_run(
    cmd: str,
    args: List[str],
    run_from: str,
    volume_mapping: Dict[str, str],
    tf_verbose: bool = True,
):
    """Run development docker image."""
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

    zipped_volume_args = zip(
        ["-v"] * len(volume_mapping),
        [f"{host}:{container}" for host, container in volume_mapping.items()],
    )

    volumes = [item for sublist in zipped_volume_args for item in sublist]

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
    print("Running the command:")
    print("\t", " ".join(docker_cmd))
    return subprocess.call(docker_cmd)


def docker_run_repo_root(
    cmd: str,
    args: List[str],
    additional_volumes: Dict[str, str] = {},
    tf_verbose: bool = True,
):
    repo_root_on_host = get_repo_root()

    docker_run(
        cmd,
        args,
        run_from=REPO_ROOT,
        volume_mapping={
            repo_root_on_host: REPO_ROOT,
            **additional_volumes,
        },
        tf_verbose=tf_verbose,
    )
