#!/usr/bin/env python3
"""One-off, provenance-gated six-model SAR reconstruction comparison.

Run from the repository root with .venv/bin/python scripts/evaluate_paper_windfields.py.
Inputs are the local conference release registry, checkpoints and effective cohorts.
A new output directory is required so failed reruns cannot leave stale metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import get_class
from torch.utils.data import DataLoader, Subset

from geo2wf.config import instantiate_datamodule, load_config_file
from geo2wf.data.collation import collate_wind_field_samples
from geo2wf.metrics.image_quality import WIND_SPEED_DATA_RANGE_MS, psnr_db_from_mse

from geo2wf.workflows._paths import REPOSITORY_ROOT as ROOT
MODELS = (
    "architecture-without-era5-unet", "architecture-with-era5-unet",
    "latent_sar_no_era5_max_wind", "latent_sar_era5_max_wind",
    "latent_sar_no_era5_max_wind_radii", "latent_sar_era5_max_wind_radii",
)


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, expected=None):
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Hash mismatch: {path}")
    return {"path": str(Path(path).resolve()), "sha256": actual}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_cohort(path):
    frame = pd.read_csv(path, keep_default_na=False)
    for column in ("sample_id", "storm_id"):
        if column not in frame or frame[column].eq("").any():
            raise ValueError(f"Missing {column}: {path}")
    if frame.sample_id.duplicated().any():
        raise ValueError(f"Duplicate sample IDs: {path}")
    return frame


def audit(registry, checkpoint_root, cohort_root):
    """Fail closed on unrecorded ancestry; collect historical exposure by storm."""
    evidence, tests, exposures = {}, {}, {}
    visiting = set()

    def visit(name):
        if name in evidence:
            return
        if name in visiting:
            raise ValueError("Checkpoint dependency cycle")
        visiting.add(name)
        entry = registry["models"][name]
        config_path, run_path = ROOT / entry["config"], ROOT / entry["run_manifest"]
        config = load_config_file(config_path)
        run = json.loads(run_path.read_text())
        dependencies = entry["dependencies"]
        # These six releases were trained from scratch. Never infer unknown ancestry.
        inputs = run.get("input_checkpoints")
        if inputs is None or any(inputs.values()):
            raise ValueError(f"Unresolved inherited checkpoint exposure: {name}: {inputs}")
        for dependency in dependencies:
            visit(dependency)
        files = {}
        frames = {}
        for split in ("training-membership", "validation", "test"):
            path = cohort_root / name / f"{split}.csv"
            files[split] = verified(path)
            frames[split] = load_cohort(path)
        for split in ("training-membership", "validation"):
            for storm in frames[split].storm_id.unique():
                exposures.setdefault(storm, set()).add(f"{name}:{split}")
        test = frames["test"]
        if not test.eligible.isin([True, False]).all():
            raise ValueError(f"Invalid eligibility flags: {name}")
        tests[name] = test.loc[test.eligible].set_index("sample_id", drop=False)
        source_files = {}
        for metadata_path in entry["data_metadata"]:
            if metadata_path.endswith(".csv.gz"):
                local = ROOT / metadata_path.removeprefix("release/provenance/").removesuffix(".gz")
                source_files[metadata_path] = verified(
                    local, registry["files"][metadata_path]["uncompressed_sha256"])
        for key in ("stats_file", "ibtracs_file"):
            spec = entry[key]
            source_files[key] = verified(ROOT / config["data"][key], spec["sha256"])
        evidence[name] = {
            "checkpoint": verified(checkpoint_root / entry["checkpoint"], entry["sha256"]),
            "config": verified(config_path, registry["files"][entry["config"]]["sha256"]),
            "run_manifest": verified(run_path, registry["files"][entry["run_manifest"]]["sha256"]),
            "input_checkpoints": inputs, "dependencies": dependencies,
            "preprocessing": config["data"], "source_files": source_files,
            "cohort_files": files, "eligible_test_samples": len(tests[name]),
        }
        visiting.remove(name)

    for name in MODELS:
        visit(name)
    common = set.intersection(*(set(tests[name].index) for name in MODELS))
    candidates = set.union(*(set(tests[name].index) for name in MODELS))
    rows = []
    for sample_id in sorted(candidates):
        records = [tests[name].loc[sample_id] for name in MODELS if sample_id in tests[name].index]
        storms = {record.storm_id for record in records}
        if len(storms) != 1:
            raise ValueError(f"Storm identity mismatch: {sample_id}")
        storm = storms.pop()
        reasons = sorted(exposures.get(storm, set()))
        if sample_id not in common:
            reasons.append("not_in_all_six_eligible_test_cohorts")
        ri = {record.is_rapid_intensification for record in records}
        if len(ri) != 1 or not ri.issubset({True, False}):
            raise ValueError(f"RI label mismatch: {sample_id}")
        # Match the underlying target asset, not just a display/sample identifier.
        targets = {json.loads(record.path_assets_json)["target_path"] for record in records}
        if len(targets) != 1:
            raise ValueError(f"Target asset mismatch: {sample_id}")
        rows.append(dict(sample_id=sample_id, storm_id=storm,
                         is_rapid_intensification=bool(ri.pop()),
                         included=not reasons, exclusion_reasons=";".join(reasons)))
    # Verify raster bytes against the release's content-addressed asset IDs.
    # Verify only used inputs; PMW is disabled in all six model configurations.
    raster_evidence = {}
    for name in MODELS:
        root = ROOT / evidence[name]["preprocessing"]["root"]
        for row in rows:
            if not row["included"]:
                continue
            record = tests[name].loc[row["sample_id"]]
            original = json.loads(record.legacy_row_json)
            assets = json.loads(record.path_assets_json)
            for key in ("condition_path", "context_path", "target_path"):
                path = root / original[key]
                identity = str(path.resolve())
                if identity not in raster_evidence:
                    raster_evidence[identity] = verified(path, assets[key])
                elif raster_evidence[identity]["sha256"] != assets[key]:
                    raise ValueError(f"Conflicting asset identity: {path}")
    evidence["verified_rasters"] = raster_evidence
    return evidence, rows


def scene_statistics(prediction, target, mask):
    """Physical-unit errors on SAR support, with no prediction-dependent masking."""
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask grids differ")
    valid = mask.bool() & torch.isfinite(target)
    if not valid.any():
        raise ValueError("No finite observed SAR pixels")
    if not torch.isfinite(prediction[valid]).all():
        raise ValueError("Nonfinite prediction on observed SAR support")
    error = prediction[valid].double() - target[valid].double()
    return {"valid_pixels": int(valid.sum()), "absolute_error_sum": error.abs().sum().item(),
            "squared_error_sum": error.square().sum().item()}


def aggregate(rows):
    count = sum(row["valid_pixels"] for row in rows)
    mse = math.fsum(row["squared_error_sum"] for row in rows) / count if count else None
    return dict(samples=len(rows), storms=len({row["storm_id"] for row in rows}),
                valid_pixels=count,
                l1_ms=math.fsum(row["absolute_error_sum"] for row in rows) / count if count else None,
                mse_ms2=mse,
                psnr_db=psnr_db_from_mse(mse) if mse is not None and mse > 0 else None,
                availability_reason="empty_subset" if not count else "perfect_reconstruction_psnr_infinite" if mse == 0 else "")


def evaluate(registry, evidence, cohort, args):
    selected = {row["sample_id"]: row for row in cohort if row["included"]}
    ids = sorted(selected)
    reference = {}
    all_rows = []
    for name in MODELS:
        print(f"Evaluating {name}: {len(ids)} scenes", flush=True)
        entry = registry["models"][name]
        config = load_config_file(ROOT / entry["config"])
        config["data"].update(batch_size=args.batch_size, num_workers=0, random_flips=False)
        for key in ("root", "stats_file", "ibtracs_file"):
            config["data"][key] = str(ROOT / config["data"][key])
        dm = instantiate_datamodule(config)
        dm.setup("test")
        dataset = dm.test_dataset
        index = {str(sid): i for i, sid in enumerate(dataset.samples.sample_id)}
        if not set(ids).issubset(index):
            raise ValueError(f"Current loader missing audited samples for {name}")
        loader = DataLoader(Subset(dataset, [index[sid] for sid in ids]),
                            batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_wind_field_samples)
        model = get_class(entry["model_target"]).load_from_checkpoint(
            evidence[name]["checkpoint"]["path"], map_location="cpu", strict=True).eval().to(args.device)
        model.validate_data_spec(dataset.data_spec)
        with torch.inference_mode():
            for batch in loader:
                device_batch = {key: value.to(args.device) if torch.is_tensor(value) else value
                                for key, value in batch.items()}
                predictions = model.predict_physical(device_batch).cpu()
                for i, sid in enumerate(batch["sample_id"]):
                    target, mask = batch["target_physical"][i], batch["target_mask"][i].bool()
                    valid = mask & torch.isfinite(target)
                    # Exact reference grid/support and RI agreement across all six loaders.
                    signature = hashlib.sha256(batch["target_bounds"][i].numpy().tobytes() +
                                               valid.numpy().tobytes() +
                                               torch.where(valid, target, 0).numpy().tobytes()).hexdigest()
                    if sid in reference and signature != reference[sid]:
                        raise ValueError(f"Target/support mismatch: {name}/{sid}")
                    reference[sid] = signature
                    if bool(batch["is_rapid_intensification"][i]) != selected[sid]["is_rapid_intensification"]:
                        raise ValueError(f"Loader RI mismatch: {sid}")
                    if batch["meta"][i]["storm_id"] != selected[sid]["storm_id"]:
                        raise ValueError(f"Loader storm mismatch: {sid}")
                    all_rows.append(dict(model=name, sample_id=sid,
                                         storm_id=selected[sid]["storm_id"],
                                         is_rapid_intensification=selected[sid]["is_rapid_intensification"],
                                         support_sha256=signature,
                                         **scene_statistics(predictions[i], target, mask)))
        del model, dm, dataset
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/paper-windfield-evaluation")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "logs/conference-release-artifacts")
    parser.add_argument("--cohort-root", type=Path, default=ROOT / "playground/conference-release/artifacts/upload/data/effective-cohorts")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    registry_path = ROOT / "release/registry.json"
    registry = json.loads(registry_path.read_text())
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="auditing",
                  registry=verified(registry_path), script=verified(Path(__file__)),
                  torch_version=torch.__version__, numpy_version=np.__version__,
                  device=args.device, batch_size=args.batch_size, threads=args.threads,
                  data_range_ms=WIND_SPEED_DATA_RANGE_MS)
    try:
        evidence, cohort = audit(registry, args.checkpoint_root, args.cohort_root)
        report["models"] = evidence
        write_csv(args.output_dir / "cohort.csv", cohort, list(cohort[0]) if cohort else
                  ["sample_id", "storm_id", "is_rapid_intensification", "included", "exclusion_reasons"])
        selected = [row for row in cohort if row["included"]]
        report["cohort"] = dict(samples=len(selected), storms=len({row["storm_id"] for row in selected}),
                                excluded_samples=len(cohort)-len(selected))
        write_json(args.output_dir / "provenance.json", report)
        if not selected:
            report["status"] = "no_common_held_out_cohort"
            return
        scenes = evaluate(registry, evidence, cohort, args)
        write_csv(args.output_dir / "per-scene.csv", scenes, list(scenes[0]))
        # Reduce the serialized export itself, so the published table is reproducible from it.
        exported = pd.read_csv(args.output_dir / "per-scene.csv").to_dict("records")
        summaries = []
        for name in MODELS:
            for subset in ("all", "ri"):
                rows = [r for r in exported if r["model"] == name and
                        (subset == "all" or r["is_rapid_intensification"])]
                summaries.append(dict(model=name, subset=subset, **aggregate(rows)))
        write_csv(args.output_dir / "metrics.csv", summaries, list(summaries[0]))
        lines = ["# Held-out SAR windfield reconstruction", "",
                 "| Model | L1 (m/s) ↓ | PSNR (dB) ↑ | RI L1 | RI PSNR |", "|---|---:|---:|---:|---:|"]
        def fmt(value):
            return "—" if value is None else f"{value:.3f}"
        for name in MODELS:
            full, ri = [s for s in summaries if s["model"] == name]
            lines.append(f"| {name} | {fmt(full['l1_ms'])} | {fmt(full['psnr_db'])} | {fmt(ri['l1_ms'])} | {fmt(ri['psnr_db'])} |")
        lines += ["", f"All: {summaries[0]['samples']} scenes, {summaries[0]['storms']} storms, {summaries[0]['valid_pixels']} pixels. "
                  f"RI: {summaries[1]['samples']} scenes, {summaries[1]['storms']} storms, {summaries[1]['valid_pixels']} pixels.", "",
                  "L1 is pooled absolute wind-speed error in m/s against finite observed SAR pixels on the "
                  "192 × 192 center crop. PSNR is 10 log10(79.8² / pooled MSE), in dB; images are not "
                  "individually normalized or clipped for scoring. All six released checkpoints use identical "
                  "test scenes and SAR support. Storms appearing in any checkpoint's training membership or "
                  "validation/model-selection cohort are excluded. Run manifests record no inherited checkpoints. "
                  "RI uses the retained 30 kt increase over the preceding 24 hours labels. "
                  "Counts and sufficient statistics are provided in the CSV exports.", ""]
        (args.output_dir / "table.md").write_text("\n".join(lines))
        report["status"] = "complete"
        report["outputs"] = {p.name: verified(p) for p in args.output_dir.iterdir() if p.name != "provenance.json"}
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(args.output_dir / "provenance.json", report)
        print(f"{report['status']}: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
