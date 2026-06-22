"""Shared filesystem locations for command-line entry points."""
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_RESULTS = REPO_ROOT / "results" / "local"


def ensure_parent_dir(path):
    """Create the parent directory for a file path when it is not the cwd."""
    parent = Path(path).expanduser().parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
