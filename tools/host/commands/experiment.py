"""Run an experiment."""
from argparse import Namespace
from pathlib import Path

from tools.host.commands.docker_commands import ContainerVolume, docker_run_repo_root

PROJECTS_DIR_RELPATH = "projects"


def start_experiment(args: Namespace):
    """Start an experiment in docker container."""
    experiment_name = args.name
    experiment_tasks = args.tasks
    experiment_dir = args.experiment_dir
    data_dir = args.data_dir
    checkpoint_to_restore = args.restore_from_chkpt
    project_args = args.project_args
    debug = args.debug

    assert Path(data_dir).is_dir()
    Path(experiment_dir).mkdir(exist_ok=True, parents=True)
    assert any(task in args.tasks.split(",") for task in ["train", "eval", "infer"])
    if checkpoint_to_restore:
        assert Path(checkpoint_to_restore)

    container_volume = ContainerVolume(experiment_dir, data_dir, checkpoint_to_restore)

    docker_run_repo_root(
        cmd="python3",
        args=[
            f"{PROJECTS_DIR_RELPATH}/{args.project}/main.py",
            "-n",
            experiment_name,
            "-t",
            experiment_tasks,
            "-d",
            ContainerVolume.DATA_DIR,
            "-e",
            ContainerVolume.EXPERIMENT_DIR,
            *(["-r", ContainerVolume.CHECKPOINT_DIR] if checkpoint_to_restore else []),
            *(["--debug"] if debug else []),
            *project_args,
        ],
        additional_volumes=container_volume.make_mapping(),
        tf_verbose=args.verbose_tf,
    )
