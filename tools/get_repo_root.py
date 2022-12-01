"""Repository root utility."""
import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_repo_root() -> str:
    """Get repository root."""
    toplevel_cmd = ["git", "rev-parse", "--show-toplevel"]
    return str(subprocess.check_output(toplevel_cmd), encoding="UTF-8").strip("\n")
