"""Validate migrated datasets against original assets and publish effective cohorts."""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

from geo2wf.workflows._paths import REPOSITORY_ROOT as ROOT
import numpy as np
import pandas as pd
import torch
import yaml
from geo2wf.config.loading import instantiate_datamodule
from geo2wf.data.catalog import ScientificCatalog, configure_release, sha256
from geo2wf.workflows.conference_release import prepare, registry
from geo2wf.workflows.release_hub import table, dump, validation_fingerprint


def compare(a, b, path="sample"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True, msg=path)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            compare(a[k], b[k], path + "." + k)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path + f"[{i}]")
    elif isinstance(a, float) and math.isnan(a):
        assert math.isnan(b), path
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def dataset(dm, split, augment=False):
    if hasattr(dm, "_make_dataset"):
        return dm._make_dataset(split, augment=augment)
    return dm._dataset(split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--work-dir", type=Path, default=Path("build/catalog-validation")
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    c = ScientificCatalog(args.root, offline=True)
    reg = registry()
    work = args.work_dir.resolve()
    prepare(
        reg, args.artifact_root.resolve(), args.source_root.resolve() / "data", work
    )
    configure_release(c, reg, work)
    mapping = pd.read_parquet(c.root / "provenance/path-mapping.parquet")
    original_assets = {
        r.asset_id: args.source_root.resolve() / r.original_path
        for r in mapping.itertuples()
    }
    report = []
    memo = {}
    for identifier, model in reg["models"].items():
        if identifier not in c.release["cohorts"]:
            continue
        spec = c.release["cohorts"][identifier]
        config_path = work / Path(model["config"]).relative_to("release/provenance")
        new_config = yaml.safe_load(config_path.read_text())
        old_config = copy.deepcopy(new_config)
        old_root = work / "original-views" / identifier
        old_root.mkdir(parents=True, exist_ok=True)
        for split, legacy in spec["legacy_splits"].items():
            records = []
            for _, row in c.cohort(identifier, split).iterrows():
                record = json.loads(row.legacy_row_json)
                for field, asset in json.loads(row.path_assets_json).items():
                    record[field] = str(original_assets[asset])
                records.append(record)
            (old_root / legacy).mkdir(exist_ok=True)
            pd.DataFrame(records).to_csv(
                old_root / legacy / "manifest.csv", index=False
            )
        for name, relative in spec["metadata"].items():
            (old_root / name).write_bytes((c.root / relative).read_bytes())
        old_config["data"]["root"] = str(old_root)
        if "stats_file" in old_config["data"]:
            old_config["data"]["stats_file"] = str(old_root / "stats.json")
        if "eligibility_cache_dir" in old_config["data"]:
            old_config["data"]["eligibility_cache_dir"] = str(old_root / ".eligibility")
        old_dm = new_dm = None
        effective_frames = {}
        used_splits = {
            new_config["data"].get(k, default)
            for k, default in [
                ("train_split", "train"),
                ("val_split", "val"),
                ("test_split", "test"),
            ]
        }
        for split, legacy in spec["legacy_splits"].items():
            if legacy not in used_splits:
                continue
            original = c.cohort(identifier, split)
            settings = {
                k: v
                for k, v in new_config["data"].items()
                if k not in ("root", "stats_file", "eligibility_cache_dir")
            }
            signature = hashlib.sha256(
                json.dumps(
                    {
                        "settings": settings,
                        "rows": original.legacy_row_json.tolist(),
                        "assets": original.path_assets_json.tolist(),
                        "metadata": {
                            k: sha256(c.root / v) for k, v in spec["metadata"].items()
                        },
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            print(f"Validating {identifier}/{split}", flush=True)
            if signature not in memo:
                if old_dm is None:
                    old_dm = instantiate_datamodule(old_config)
                    new_dm = instantiate_datamodule(new_config)
                old, new = dataset(old_dm, legacy), dataset(new_dm, legacy)
                assert list(old.samples.sample_id) == list(new.samples.sample_id)
                for index in range(len(old)):
                    compare(old[index], new[index])
                # Reuse loaded cohorts: changing augmentation does not change membership.
                if hasattr(old_dm, "_make_dataset") and len(old):
                    a = getattr(old, "paired_dataset", getattr(old, "paired", old))
                    b = getattr(new, "paired_dataset", getattr(new, "paired", new))
                    a.augment = b.augment = True
                    torch.manual_seed(123)
                    np.random.seed(123)
                    x = old[0]
                    torch.manual_seed(123)
                    np.random.seed(123)
                    compare(x, new[0])
                    a.augment = b.augment = False
                labels = getattr(new, "_labels", getattr(new, "labels", None))
                memo[signature] = {
                    "ids": list(new.samples.sample_id),
                    "ri": (
                        [r.get("is_rapid_intensification") for r in labels]
                        if labels is not None
                        else None
                    ),
                    "storms": int(new.samples.storm_id.nunique()),
                }
            verified = memo[signature]
            lookup = original.set_index("sample_id", drop=False)
            effective = lookup.loc[verified["ids"]].copy().reset_index(drop=True)
            effective["ordinal"] = range(len(effective))
            effective["eligible"] = True
            effective_frames[legacy] = effective
            if verified["ri"] is not None:
                effective["is_rapid_intensification"] = verified["ri"]
            relative = f"effective-cohorts/{identifier}/{split}.parquet"
            table(c.root, relative, effective.to_dict("records"))
            c.release["cohorts"][identifier].setdefault("effective_splits", {})[
                split
            ] = relative
            eligible = set(verified["ids"])
            original["eligible"] = original.sample_id.isin(eligible)
            table(c.root, spec["splits"][split], original.to_dict("records"))
            report.append(
                {
                    "experiment": identifier,
                    "split": split,
                    "source_rows": len(original),
                    "effective_rows": len(effective),
                    "exact_sample_comparisons": len(effective),
                    "comparison_signature": signature,
                    "storms": verified["storms"],
                }
            )
        training_split = new_config["data"].get("train_split", "train")
        if training_split in effective_frames:
            components = [effective_frames[training_split]]
            if new_config["data"].get("include_test_in_train", False):
                components.append(
                    effective_frames[new_config["data"].get("test_split", "test")]
                )
            training = pd.concat(components, ignore_index=True)
            training["original_split"] = training["split"]
            training["split"] = "train"
            training["ordinal"] = range(len(training))
            relative = f"effective-cohorts/{identifier}/training-membership.parquet"
            table(c.root, relative, training.to_dict("records"))
            c.release["cohorts"][identifier]["training_membership"] = relative
            c.release["cohorts"][identifier]["include_test_in_train"] = new_config[
                "data"
            ].get("include_test_in_train", False)
    configs = yaml.safe_load((c.root / "README.md").read_text().split("---")[1])
    configs["configs"] = [
        v for v in configs["configs"] if not v["config_name"].endswith("-effective")
    ]
    for name, spec in c.release["cohorts"].items():
        if "effective_splits" in spec:
            configs["configs"].append(
                {
                    "config_name": name + "-effective",
                    "data_files": [
                        {"split": split, "path": path}
                        for split, path in spec["effective_splits"].items()
                    ],
                }
            )
    body = (c.root / "README.md").read_text().split("---", 2)[2]
    (c.root / "README.md").write_text(
        "---\n" + yaml.safe_dump(configs, sort_keys=False) + "---" + body
    )
    dump(c.root / "validation/loaders.json", report)
    c.release["files"] = {
        str(p.relative_to(c.root)): sha256(p)
        for p in c.root.rglob("*")
        if p.is_file()
        and not str(p.relative_to(c.root)).startswith("assets/")
        and p.name != "release.json"
    }
    c.release["loader_validation"] = {
        "fingerprint": validation_fingerprint(c.root),
        "report_sha256": sha256(c.root / "validation/loaders.json"),
    }
    dump(c.root / "release.json", c.release)
    dump(ROOT / "release/hosting/validation.json", report)
    lines = [
        "# Published dataset cohorts",
        "",
        "Generated from the effective catalog cohorts. Source manifests and filtered memberships are both published. Counts below describe individual splits; training-membership indexes additionally record historical test reuse where configured.",
        "",
        "| Experiment | Split | Samples | Storms |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f'| {r["experiment"]} | {r["split"]} | {r["effective_rows"]} | {r["storms"]} |'
        for r in report
    ]
    (ROOT / "docs/data/published-cohorts.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
