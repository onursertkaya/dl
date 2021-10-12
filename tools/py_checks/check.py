"""Wrapper for running checkjobs in Docker container."""
from functools import lru_cache
from pathlib import Path
from typing import List

from tools.docker.commands import REPO_ROOT, docker_run_repo_root
from tools.get_repo_root import get_repo_root

PYLINT_CONFIG_RELPATH = "tools/py_checks/config/pylintrc"
PYCODESTYLE_CONFIG_RELPATH = "tools/py_checks/config/pycodestyle.cfg"


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
        )
        print(f"\n=== finished running {job}\n")

    _run_pylint()


def run_py_tests():
    """Discover and run all pythen tests."""
    # test discovery without __init__.py files does not work before python3.11
    test_files = filter(lambda path: "test" in path, _get_py_files_with_docker_paths())

    docker_run_repo_root(
        cmd="python3",
        args=[
            f"-m",
            "unittest",
            *test_files,
        ],
        tf_verbose=False,
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
    )
    print(f"\n=== finished running pylint\n")


@lru_cache(maxsize=1)
def _get_py_files_with_docker_paths() -> List[str]:
    repo_root_path = Path(get_repo_root())
    return sorted(
        map(
            lambda path: str(path.relative_to(repo_root_path)),
            repo_root_path.rglob("*.py"),
        )
    )
