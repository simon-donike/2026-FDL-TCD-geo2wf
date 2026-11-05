"""Compatibility wrapper for :mod:`geo2wf.workflows.enrich_dataset_manifests`."""

from geo2wf.workflows.enrich_dataset_manifests import *  # noqa: F401,F403
from geo2wf.workflows import enrich_dataset_manifests as _workflow

globals().update(
    {name: value for name, value in vars(_workflow).items() if not name.startswith("__")}
)

import sys as _sys

_sys.modules[__name__] = _workflow

if __name__ == "__main__" and callable(getattr(_workflow, "main", None)):
    _workflow.main()
