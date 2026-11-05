"""Evaluate model predictions."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "conference":
        from geo2wf.workflows.conference_release import main as evaluate

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        evaluate()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        choices=(
            "conference",
            "intensity-comparison",
            "intensity-correction",
            "intensity-forecast",
        ),
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.argv = [sys.argv[0], *remaining]
    if args.workflow == "conference":
        from geo2wf.workflows.conference_release import main as evaluate
    elif args.workflow == "intensity-comparison":
        from geo2wf.workflows.evaluate_intensity_models import main as evaluate
    elif args.workflow == "intensity-correction":
        from geo2wf.workflows.evaluate_intensity_correction import main as evaluate
    else:
        from geo2wf.workflows.evaluate_intensity_forecast import main as evaluate
    evaluate()
