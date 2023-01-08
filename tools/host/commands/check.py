"""Wrapper for running checkjobs in Docker container."""
from tools.host.commands.docker_commands import ContainerVolume, docker_run_repo_root
from tools.host.constants import Colors
from tools.host.misc import get_py_files_with_docker_paths

CONF_ROOT = "tools/conf"
PYLINT_CONFIG_RELPATH = f"{CONF_ROOT}/pylintrc"
PYCODESTYLE_CONFIG_RELPATH = f"{CONF_ROOT}/pycodestyle.cfg"
MYPY_CONFIG_RELPATH = f"{CONF_ROOT}/mypy.ini"


def run_py_checks():
    """Run all python checks for docs, formatting and PEP."""
    for job, job_args in (
        ("isort", ["--profile", "black"]),
        ("black", []),
        ("pycodestyle", [f"--config={PYCODESTYLE_CONFIG_RELPATH}"]),
        ("pydocstyle", []),
    ):
        docker_run_repo_root(
            "python3",
            args=["-m", job, *job_args, ContainerVolume.REPO_ROOT],
            continue_running=True,
        )
        print(f"{Colors.BLUE}\n=== finished running {job}{Colors.RESET}\n")

    _run_pylint()
    _run_mypy()


def _run_pylint():
    docker_run_repo_root(
        cmd="python3",
        args=[
            "-m",
            "pylint",
            f"--rcfile={PYLINT_CONFIG_RELPATH}",
            *get_py_files_with_docker_paths(),
        ],
        continue_running=True,
    )
    print("\n=== finished running pylint\n")


def _run_mypy():
    docker_run_repo_root(
        cmd="python3",
        args=[
            "-m",
            "mypy",
            "--explicit-package-bases",
            "--config-file",
            MYPY_CONFIG_RELPATH,
            *get_py_files_with_docker_paths(),
        ],
    )
    print("\n=== finished running mypy\n")
