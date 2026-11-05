"""Run one of the maintained storm-inference workflows."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workflow",
        choices=(
            "deterministic-residual",
            "intensity-correction",
            "intensity-forecast",
        ),
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.argv = [sys.argv[0], *remaining]
    if args.workflow == "deterministic-residual":
        from geo2wf.workflows.run_storm_unet_inference import main as inference_main
    elif args.workflow == "intensity-correction":
        from geo2wf.workflows.run_intensity_correction_inference import main as inference_main
    else:
        from geo2wf.workflows.run_intensity_forecast_inference import main as inference_main
    inference_main()
