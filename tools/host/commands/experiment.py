"""Run an experiment."""
from pathlib import Path

from core.settings import ExperimentSettings
from tools.host.commands.docker_commands import docker_run_repo_root

PROJECTS_DIR_RELPATH = "projects"


def start_experiment(project: str, verbose_tf: bool = True):
    """Start an experiment in docker container."""
    settings = ExperimentSettings.load_project_default(project)
    _ensure_paths(settings)

    docker_run_repo_root(
        cmd="python3",
        args=[
            f"{PROJECTS_DIR_RELPATH}/{project}/main.py",
        ],
        additional_volumes=_make_volume_mapping(settings),
        tf_verbose=verbose_tf,
    )


def _ensure_paths(settings: ExperimentSettings):
    Path(settings.output_directory).mkdir(exist_ok=False, parents=True)
    assert Path(settings.data_directory).is_dir()
    if settings.restore_from is not None:
        assert Path(settings.restore_from).is_dir()


def _make_volume_mapping(settings: ExperimentSettings):
    mapping = {
        settings.output_directory: settings.output_directory,
        settings.data_directory: settings.data_directory,
    }
    return mapping
