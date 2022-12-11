"""Miscellaneous utilities."""
from functools import lru_cache
from pathlib import Path
from typing import List

from tools.get_repo_root import get_repo_root


@lru_cache(maxsize=1)
def get_py_files_with_docker_paths() -> List[str]:
    """Get python file paths adjusted to be in the container."""
    repo_root_path = Path(get_repo_root())
    return sorted(
        map(
            lambda path: str(path.relative_to(repo_root_path)),
            repo_root_path.rglob("*.py"),
        )
    )
