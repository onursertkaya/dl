"""Wrapper for running tests in Docker container."""
from tools.commands.docker_commands import docker_run_repo_root
from tools.misc import get_py_files_with_docker_paths


def run_py_tests():
    """Discover and run all python tests."""
    # test discovery without __init__.py files does not work before python3.11
    test_files = filter(lambda path: "test" in path, get_py_files_with_docker_paths())

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
