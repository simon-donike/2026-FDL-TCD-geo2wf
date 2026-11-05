"""Repository paths used by workflow modules."""

import os
from pathlib import Path


def repository_root() -> Path:
    configured = os.environ.get("GEO2WF_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    package_candidate = Path(__file__).resolve().parents[3]
    if (package_candidate / "pyproject.toml").exists():
        return package_candidate
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


REPOSITORY_ROOT = repository_root()
