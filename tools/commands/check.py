"""Wrapper for running checkjobs in Docker container."""
from functools import lru_cache
from pathlib import Path
from typing import List

from tools.commands.docker_commands import REPO_ROOT, docker_run_repo_root
from tools.get_repo_root import get_repo_root

PYLINT_CONFIG_RELPATH = "tools/conf/pylintrc"
PYCODESTYLE_CONFIG_RELPATH = "tools/conf/pycodestyle.cfg"
MYPY_CONFIG_RELPATH = "tools/conf/mypy.ini"


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
            args=["-m", job, *job_args, REPO_ROOT],
            continue_running=True,
        )
        print(f"\n=== finished running {job}\n")

    _run_pylint()
    _run_mypy()


def run_py_tests():
    """Discover and run all python tests."""
    # test discovery without __init__.py files does not work before python3.11
    test_files = filter(lambda path: "test" in path, _get_py_files_with_docker_paths())

    docker_run_repo_root(
        cmd="python3",
        args=[
            "-m",
            "unittest",
            *test_files,
        ],
        tf_verbose=False,
        continue_running=True,
    )


def _run_pylint():
    docker_run_repo_root(
        cmd="python3",
        args=[
            "-m",
            "pylint",
            f"--rcfile={PYLINT_CONFIG_RELPATH}",
            *_get_py_files_with_docker_paths(),
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
            *_get_py_files_with_docker_paths(),
        ],
    )
    print("\n=== finished running mypy\n")


@lru_cache(maxsize=1)
def _get_py_files_with_docker_paths() -> List[str]:
    repo_root_path = Path(get_repo_root())
    return sorted(
        map(
            lambda path: str(path.relative_to(repo_root_path)),
            repo_root_path.rglob("*.py"),
        )
    )
