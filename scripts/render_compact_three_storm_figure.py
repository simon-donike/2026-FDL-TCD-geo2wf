"""Compatibility wrapper for :mod:`geo2wf.workflows.render_compact_three_storm_figure`."""

from geo2wf.workflows.render_compact_three_storm_figure import *  # noqa: F401,F403
from geo2wf.workflows import render_compact_three_storm_figure as _workflow

globals().update(
    {name: value for name, value in vars(_workflow).items() if not name.startswith("__")}
)

import sys as _sys

_sys.modules[__name__] = _workflow

if __name__ == "__main__" and callable(getattr(_workflow, "main", None)):
    _workflow.main()
