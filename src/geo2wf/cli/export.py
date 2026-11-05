"""Run one of the maintained dataset export workflows."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        choices=(
            "geo-sar",
            "intensity-cache",
            "joint-intensity-cache",
            "intensity-forecast-cache",
        ),
    )
    args = parser.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    sys.argv = [sys.argv[0], *remaining]
    if args.dataset == "geo-sar":
        from geo2wf.workflows.export_geo_sar_geotiffs import main as export_main
    elif args.dataset == "intensity-cache":
        from geo2wf.workflows.export_unet_intensity_cache import main as export_main
    elif args.dataset == "joint-intensity-cache":
        from geo2wf.workflows.export_joint_intensity_cache import main as export_main
    else:
        from geo2wf.workflows.export_intensity_forecast_cache import main as export_main
    export_main()
