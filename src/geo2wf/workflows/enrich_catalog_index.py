"""Add queryable scientific metadata to an existing catalog without touching assets."""

import argparse
import json
from pathlib import Path
import sys

from geo2wf.workflows._paths import REPOSITORY_ROOT as ROOT
import pandas as pd
import pyarrow.parquet as pq
from geo2wf.workflows.release_hub import table, dump, refresh_checksums


def enrich(root):
    samples = pd.read_parquet(root / "index/samples.parquet")
    metadata = samples.metadata_json.map(json.loads)
    for name in (
        "center_lat",
        "center_lon",
        "ibtracs_msw_ms",
        "ibtracs_msw_kt",
        "ibtracs_mslp_hpa",
        "target_wind_ms",
        "target_delta_ms",
        "dt_minutes",
        "pmw_dt_minutes",
        "grid_resolution",
    ):
        samples[name] = metadata.map(
            lambda row: float(row[name]) if row.get(name) not in (None, "") else None
        )
    for name in ("geo_channels", "sar_channels", "era5_channels", "pmw_channels"):
        samples[name] = metadata.map(
            lambda row: json.loads(row[name]) if row.get(name) else []
        )
    table(root, "index/samples.parquet", samples.to_dict("records"))
    obs = pd.read_parquet(root / "index/observations.parquet")
    obs["observation_key"] = obs.apply(
        lambda row: str(row.observation_id)
        + ":"
        + str(row.asset_id)
        + ":"
        + str(row.case_study),
        axis=1,
    )
    obs["representation"] = obs.case_study.map({True: "source", False: "processed"})
    obs["upstream_source_included"] = obs.case_study
    metadata = obs.metadata_json.map(json.loads)
    for name in ("center_lat", "center_lon", "height", "width"):
        obs[name] = metadata.map(
            lambda row: float(row[name]) if row.get(name) not in (None, "") else None
        )
    obs["variables"] = metadata.map(
        lambda row: json.loads(row["variables"]) if row.get("variables") else []
    )
    table(root, "index/observations.parquet", obs.to_dict("records"))
    storms = pd.read_parquet(root / "index/storms.parquet")
    original = root / "provenance/original-indexes/storms/spacehpc_manifest_splits.csv"
    source = pd.read_csv(original, keep_default_na=False).set_index("storm_id")
    storms["original_index_json"] = storms.storm_id.map(
        lambda sid: (
            json.dumps(source.loc[sid].to_dict()) if sid in source.index else None
        )
    )
    table(root, "index/storms.parquet", storms.to_dict("records"))
    schema = json.loads((root / "schema.json").read_text())
    schema["tables"] = {
        p.stem: str(pq.read_schema(p)) for p in (root / "index").glob("*.parquet")
    }
    dump(root / "schema.json", schema)
    refresh_checksums(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    enrich(parser.parse_args().root)
