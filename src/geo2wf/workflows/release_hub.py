"""Prepare, verify and download the indexed conference dataset; upload explicitly."""

from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

from geo2wf.workflows._paths import REPOSITORY_ROOT as ROOT
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from geo2wf.data.catalog import ScientificCatalog, sha256, safe_path


def slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    csv.field_size_limit(100_000_000)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def table(root, relative, records):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    for col in frame:
        if col.endswith("_timestamp") or col == "timestamp":
            frame[col] = pd.to_datetime(
                frame[col], utc=True, errors="raise", format="mixed"
            )
    arrow = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(arrow, path, compression="zstd")
    # JSON-encode list/dict cells in the spreadsheet export.
    for col in frame:
        frame[col] = frame[col].map(
            lambda v: (
                json.dumps(v.tolist() if hasattr(v, "tolist") else v)
                if isinstance(v, (list, dict)) or hasattr(v, "tolist")
                else v
            )
        )
    frame.to_csv(path.with_suffix(".csv"), index=False)


class Builder:
    def __init__(self, source, output):
        self.source, self.output = source.resolve(), output.resolve()
        self.assets, self.by_source, self.samples, self.mapping = {}, {}, {}, []
        self.cohorts, self.observations = {}, []
        self.gaps = []
        self.output.mkdir(parents=True, exist_ok=True)

    def copy(self, source, relative):
        dest = safe_path(self.output, relative)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or sha256(dest) != sha256(source):
            temporary = dest.with_name(dest.name + ".partial")
            shutil.copyfile(source, temporary)
            temporary.replace(dest)
        return relative

    def asset(self, source, relative, *, storm="", profiles=(), producer=""):
        source = Path(source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        key = self.by_source.get(str(source))
        if not key:
            key = sha256(source)
            self.by_source[str(source)] = key
        if key not in self.assets:
            self.copy(source, relative)
            entry = {
                "asset_id": key,
                "sha256": key,
                "path": relative,
                "size_bytes": source.stat().st_size,
                "format": source.suffix.lstrip("."),
                "storm_ids": [],
                "profiles": [],
                "producer_ids": [],
                "channels": [],
                "units": [],
                "shape": [],
                "crs": None,
            }
            if source.suffix == ".tif":
                import rasterio

                with rasterio.open(source) as raster:
                    entry.update(
                        channels=list(raster.descriptions),
                        units=list(raster.units),
                        shape=[raster.count, raster.height, raster.width],
                        crs=str(raster.crs),
                        transform=list(raster.transform),
                        nodata=(
                            raster.nodata
                            if raster.nodata is None
                            or abs(raster.nodata) != float("inf")
                            else None
                        ),
                    )
            self.assets[key] = entry
        entry = self.assets[key]
        for field, values in [
            ("storm_ids", [storm] if storm else []),
            ("profiles", profiles),
            ("producer_ids", [producer] if producer else []),
        ]:
            entry[field] = sorted(set(entry[field]) | set(values))
        try:
            old = str(source.relative_to(self.source))
        except ValueError:
            old = source.name
        self.mapping.append(
            {"original_path": old, "asset_id": key, "path": entry["path"]}
        )
        return key

    def cohort(
        self, identifier, data_root, manifests, *, metadata=None, experiment=None
    ):
        print(f"Indexing {identifier}", flush=True)
        version = slug(str(data_root.relative_to(self.source)))
        spec = {
            "processing_version": version,
            "splits": {},
            "legacy_splits": {},
            "metadata": {},
            "support_asset_ids": [],
            "experiment_id": experiment,
            "selection": "original manifest; original runtime eligibility filters still apply",
        }
        for name, path in (metadata or {}).items():
            dest = (
                f"normalization/{identifier}/{name}"
                if name == "stats.json"
                else f"provenance/preprocessing/{identifier}/{name}"
            )
            spec["metadata"][name] = self.copy(path, dest)
        for manifest in manifests:
            legacy_split = manifest.parent.name
            split = "validation" if legacy_split == "val" else legacy_split
            entries = []
            original = f"provenance/original-indexes/{identifier}/{legacy_split}/{manifest.name}"
            self.copy(manifest, original)
            for ordinal, row in enumerate(rows(manifest)):
                sid = row.get("sample_id") or f"{legacy_split}-{ordinal}"
                storm = row.get("storm_id", "")
                paths = {}
                profiles = ["train"] if "train" in split else ["paper-eval", "train"]
                if identifier.startswith("dashboard"):
                    profiles.append("storms")
                for column, value in row.items():
                    if not column.endswith("_path") or not value:
                        continue
                    source = Path(value)
                    if not source.is_absolute():
                        source = data_root / value
                    if source.suffix not in (".tif", ".npz", ".nc"):
                        continue
                    family = "derived" if source.suffix == ".npz" else "paired"
                    relative = f'assets/{family}/{version}/{slug(storm)}/{slug(sid)}/{column.removesuffix("_path")}{source.suffix}'
                    paths[column] = self.asset(
                        source,
                        relative,
                        storm=storm,
                        profiles=profiles,
                        producer=version,
                    )
                # Different exports of the same sample can carry different pixels.
                fingerprint = hashlib.sha256(
                    json.dumps(paths, sort_keys=True).encode()
                ).hexdigest()[:12]
                sample_key = f"{version}:{sid}:{fingerprint}"
                if sample_key not in self.samples:
                    sample = {
                        "sample_key": sample_key,
                        "sample_id": sid,
                        "processing_version": version,
                        "storm_id": storm,
                        "asset_ids": sorted(set(paths.values())),
                        "metadata_json": json.dumps(
                            {k: v for k, v in row.items() if not k.endswith("_path")}
                        ),
                    }
                    for modality in ("geo", "sar", "era5", "pmw"):
                        sample[f"{modality}_asset_id"] = paths.get(f"{modality}_path")
                        sample[f"has_{modality}"] = f"{modality}_path" in paths
                    for col in (
                        "geo_timestamp",
                        "sar_timestamp",
                        "era5_timestamp",
                        "pmw_timestamp",
                    ):
                        sample[col] = row.get(col) or None
                    self.samples[sample_key] = sample
                entries.append(
                    {
                        "sample_key": sample_key,
                        "sample_id": sid,
                        "storm_id": storm,
                        "ordinal": ordinal,
                        "split": split,
                        "experiment_id": experiment,
                        "asset_ids": sorted(set(paths.values())),
                        "legacy_row_json": json.dumps(row),
                        "path_assets_json": json.dumps(paths),
                        "eligible": None,
                        "is_rapid_intensification": {
                            "true": True,
                            "false": False,
                            "1": True,
                            "0": False,
                        }.get(str(row.get("is_rapid_intensification", "")).lower()),
                        "source_manifest": original,
                    }
                )
            dest = f"cohorts/{identifier}/{split}.parquet"
            table(self.output, dest, entries)
            spec["splits"][split], spec["legacy_splits"][split] = dest, legacy_split
        self.cohorts[identifier] = spec

    def finish(self, registry):
        table(self.output, "index/assets.parquet", list(self.assets.values()))
        for sample in self.samples.values():
            meta = json.loads(sample["metadata_json"])
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
                value = meta.get(name)
                sample[name] = float(value) if value not in (None, "") else None
            for name in (
                "geo_channels",
                "sar_channels",
                "era5_channels",
                "pmw_channels",
            ):
                sample[name] = json.loads(meta[name]) if meta.get(name) else []
        table(self.output, "index/samples.parquet", list(self.samples.values()))
        table(self.output, "index/observations.parquet", self.observations)
        sample_counts = Counter(s["storm_id"] for s in self.samples.values())
        observation_counts = Counter(o["storm_id"] for o in self.observations)
        storms = sorted(sample_counts.keys() | observation_counts.keys())
        table(
            self.output,
            "index/storms.parquet",
            [
                {
                    "storm_id": s,
                    "sample_count": sample_counts[s],
                    "observation_count": observation_counts[s],
                    "case_study": s in ("AL082025", "EP112025", "EP182023"),
                }
                for s in storms
            ],
        )
        table(
            self.output,
            "provenance/path-mapping.parquet",
            list({r["original_path"]: r for r in self.mapping}.values()),
        )
        dump(
            self.output / "schema.json",
            {
                "schema_version": 1,
                "tables": {
                    p.stem: str(pq.read_schema(p))
                    for p in (self.output / "index").glob("*.parquet")
                },
                "extensions": "metadata_json preserves original string values; legacy_row_json preserves original loader values.",
                "identity": "Processing version + historical sample ID + asset mapping fingerprint. Asset ID is SHA-256.",
                "cohort_semantics": "Ordered source membership. eligible=null means runtime model filtering is required; never assume matched cohorts.",
            },
        )
        configs = [
            {
                "config_name": n,
                "data_files": [{"split": "catalog", "path": f"index/{n}.parquet"}],
                "default": n == "samples",
            }
            for n in ("samples", "assets", "observations", "storms")
        ]
        for name, spec in self.cohorts.items():
            configs.append(
                {
                    "config_name": name,
                    "data_files": [
                        {"split": s, "path": p} for s, p in spec["splits"].items()
                    ],
                }
            )
        (self.output / "README.md").write_text(
            "---\n"
            + yaml.safe_dump({"configs": configs}, sort_keys=False)
            + "---\n\n"
            + (ROOT / "release/hosting/dataset-card.md").read_text()
        )
        files = {
            str(p.relative_to(self.output)): sha256(p)
            for p in sorted(self.output.rglob("*"))
            if p.is_file()
            and not str(p.relative_to(self.output)).startswith("assets/")
            and p.name != "release.json"
        }
        dump(
            self.output / "release.json",
            {
                "schema_version": 1,
                "source_commit": registry["source_commit"],
                "cohorts": self.cohorts,
                "files": files,
                "gaps": self.gaps,
                "profiles": ["index-only", "paper-eval", "train", "storms", "all"],
            },
        )


def prepare(args):
    source, output = args.source_root.resolve(), args.output.resolve()
    registry = json.loads((ROOT / "release/registry.json").read_text())
    for relative, info in registry["files"].items():
        if sha256(ROOT / relative) != info["sha256"]:
            raise ValueError(f"Original release provenance changed: {relative}")
    b = Builder(source, output / "data")
    for identifier, model in registry["models"].items():
        if not model.get("config"):
            b.gaps.append(f"{identifier}: original configuration unavailable")
            continue
        config = yaml.safe_load((ROOT / model["config"]).read_text())["data"]
        data_root = Path(config["root"])
        if data_root.is_absolute():
            data_root = source / data_root.relative_to(registry["source_root"])
        else:
            data_root = source / data_root
        manifests = [
            ROOT / p
            for p in model.get("data_metadata", [])
            if p.endswith("manifest.csv.gz")
        ]
        if not manifests:
            manifests = sorted(data_root.glob("*/manifest.csv"))
        metadata = {}
        for name in ("stats.json", "cache-metadata.json"):
            candidates = [
                ROOT / p
                for p in model.get("data_metadata", [])
                if p.endswith("/" + name)
            ]
            if candidates:
                metadata[name] = candidates[0]
            elif (data_root / name).is_file():
                metadata[name] = data_root / name
        b.cohort(
            identifier, data_root, manifests, metadata=metadata, experiment=identifier
        )
        ibtracs = config.get("ibtracs_file")
        if ibtracs:
            path = Path(ibtracs)
            path = source / (
                path.relative_to(registry["source_root"])
                if path.is_absolute()
                else path
            )
            key = b.asset(
                path,
                f"labels/ibtracs/{path.name}",
                profiles=["train", "paper-eval", "storms"],
            )
            b.cohorts[identifier]["support_asset_ids"].append(key)
            b.cohorts[identifier]["ibtracs_asset_id"] = key
    # Publish full current corpus as a separately named inventory, not a paper cohort.
    for data_root in sorted((source / "data/geotiff").glob("*")):
        if data_root.name not in (
            "geo_sar_10bands_era5",
            "geo_sar_10bands_era5_v2_pmw",
        ):
            continue
        for original in sorted(data_root.glob("*")):
            if original.is_file() and original.suffix in (".csv", ".json"):
                b.copy(
                    original,
                    f"provenance/original-indexes/inventory-{data_root.name}/{original.name}",
                )
        b.cohort(
            "inventory-" + data_root.name,
            data_root,
            sorted(data_root.glob("*/manifest.csv")),
            metadata={"stats.json": data_root / "stats.json"},
        )
    index_root = source / "inference/inf_data/index-files"
    for path in sorted(index_root.glob("*.csv")):
        b.copy(path, "provenance/original-indexes/storms/" + path.name)
    # Carry observation identities from all paired exports as well as storm sources.
    for sample in b.samples.values():
        metadata = json.loads(sample["metadata_json"])
        for modality in ("geo", "sar", "era5", "pmw"):
            oid = metadata.get(modality + "_observation_id")
            if oid:
                b.observations.append(
                    {
                        "observation_id": oid,
                        "storm_id": sample["storm_id"],
                        "timestamp": metadata.get(modality + "_timestamp") or None,
                        "source_type": modality,
                        "sensor": metadata.get(modality + "_sensor"),
                        "product": None,
                        "asset_id": sample.get(modality + "_asset_id"),
                        "available": sample.get(modality + "_asset_id") is not None,
                        "case_study": False,
                        "metadata_json": json.dumps(metadata),
                    }
                )
    local_files = {}
    for path in (source / "inference/inf_data").rglob("*"):
        if path.is_file():
            local_files.setdefault(path.name, []).append(path)
    observations = rows(index_root / "observation_manifest_v6.csv")
    for row in observations:
        relative = row["path"]
        candidates = [
            source / relative,
            source / "inference/inf_data" / relative,
            source
            / "inference/inf_data"
            / row.get("source", "")
            / row.get("filename", ""),
        ]
        # Original index may record HPC paths; match basename within the local source tree.
        matches = local_files.get(Path(relative).name, [])
        if len(matches) > 1 and len({sha256(p) for p in matches}) > 1:
            raise ValueError(f"Ambiguous source filename: {relative}")
        candidates.extend(matches)
        local = next((p for p in candidates if p.is_file()), None)
        key = None
        if local:
            key = b.asset(
                local,
                f'assets/sources/{slug(row["source_type"])}/{slug(row["storm_id"])}/{slug(row["observation_id"])}{local.suffix}',
                storm=row["storm_id"],
                profiles=["storms"],
            )
        b.observations.append(
            {
                "observation_id": row["observation_id"],
                "storm_id": row["storm_id"],
                "timestamp": row["timestamp"] or None,
                "source_type": row["source_type"],
                "sensor": row.get("sensor"),
                "product": row.get("product"),
                "asset_id": key,
                "available": key is not None,
                "case_study": True,
                "metadata_json": json.dumps(
                    {k: v for k, v in row.items() if k != "path"}
                ),
            }
        )
    # Reference results and original provenance are independently downloadable.
    for folder in (
        "release/paper",
        "release/provenance",
        "docs/assets/data/final-results",
    ):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file():
                prefix = {
                    "release/paper": "reference-results/paper",
                    "release/provenance": "provenance/experiments",
                    "docs/assets/data/final-results": "reference-results/case-studies",
                }[folder]
                b.copy(path, prefix + "/" + str(path.relative_to(ROOT / folder)))
    for path in sorted((source / "inference").rglob("*")):
        if (
            path.is_file()
            and "inf_data" not in path.parts
            and "__pycache__" not in path.parts
        ):
            relative = str(path.relative_to(source / "inference"))
            b.asset(
                path, "reference-results/dashboard/" + relative, profiles=["storms"]
            )
    b.observations = list(
        {
            (o["observation_id"], o["asset_id"], o["case_study"]): o
            for o in b.observations
        }.values()
    )
    b.finish(registry)
    from geo2wf.workflows.enrich_catalog_index import enrich

    enrich(b.output)
    models = output / "models"
    models.mkdir(exist_ok=True)
    for identifier, model in registry["models"].items():
        original = args.artifact_root / model["checkpoint"]
        if sha256(original) != model["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {identifier}")
        dest = models / model["checkpoint"]
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, dest)
    shutil.copytree(ROOT / "release", models / "release", dirs_exist_ok=True)
    shutil.copytree(
        args.artifact_root / "run-provenance",
        models / "run-provenance",
        dirs_exist_ok=True,
    )
    (models / "README.md").write_text(
        (ROOT / "release/hosting/models-card.md").read_text()
    )
    dump(
        models / "dataset-links.json",
        {
            "dataset_revision": None,
            "experiments": {
                k: {
                    "cohort": k if k in b.cohorts else None,
                    "normalization": b.cohorts.get(k, {})
                    .get("metadata", {})
                    .get("stats.json"),
                }
                for k in registry["models"]
            },
        },
    )
    print(
        json.dumps(
            {
                "assets": len(b.assets),
                "samples": len(b.samples),
                "cohorts": len(b.cohorts),
                "observations": len(b.observations),
                "gaps": b.gaps,
            },
            indent=2,
        )
    )


def refresh_checksums(root):
    release = json.loads((root / "release.json").read_text())
    release["files"] = {
        str(p.relative_to(root)): sha256(p)
        for p in root.rglob("*")
        if p.is_file()
        and not str(p.relative_to(root)).startswith(("assets/", ".cache/"))
        and p.name not in ("release.json", "SHA256SUMS")
    }
    dump(root / "release.json", release)


def validation_fingerprint(root):
    paths = [root / "index/assets.parquet"]
    for directory in ("cohorts", "effective-cohorts", "normalization"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file())
    encoded = json.dumps(
        {str(p.relative_to(root)): sha256(p) for p in sorted(paths)}, sort_keys=True
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def upload(args):
    # Deliberately separate from prepare: this action performs external writes.
    from huggingface_hub import HfApi, hf_hub_download

    config = json.loads(args.publish_config.read_text())
    for field in (
        "dataset_repo",
        "model_repo",
        "attribution",
    ):
        if (
            not config.get(field)
            or "REPLACE" in config[field]
            or "YOUR-ORG" in config[field]
        ):
            raise ValueError(f"Configure {field} before publishing")
    for field in ("dataset_license", "model_license"):
        if config.get(field) and "REPLACE" in config[field]:
            raise ValueError(f"Configure {field} or use null for unspecified licensing")
    print("Validating staged scientific assets and checkpoints", flush=True)
    root = args.output.resolve()
    catalog = ScientificCatalog(root / "data", offline=True)
    catalog.verify()
    validation = catalog.release.get("loader_validation", {})
    if validation.get("fingerprint") != validation_fingerprint(root / "data"):
        raise ValueError(
            "Run validate_catalog_loaders.py on this exact package before publishing"
        )
    if validation.get("report_sha256") != sha256(root / "data/validation/loaders.json"):
        raise ValueError("Loader validation report has changed")
    registry = json.loads((root / "models/release/registry.json").read_text())
    for relative, info in registry["files"].items():
        if sha256(root / "models" / relative) != info["sha256"]:
            raise ValueError(f"Original model provenance changed: {relative}")
    for model in registry["models"].values():
        if sha256(root / "models" / model["checkpoint"]) != model["sha256"]:
            raise ValueError("Model checkpoint differs from the release registry")
    api = HfApi()
    revisions = {}
    for name, kind, repo_field, license_field in [
        ("data", "dataset", "dataset_repo", "dataset_license"),
        ("models", "model", "model_repo", "model_license"),
    ]:
        folder = root / name
        readme = (folder / "README.md").read_text()
        if readme.startswith("---"):
            _, front, body = readme.split("---", 2)
            front = yaml.safe_load(front)
        else:
            front, body = {}, "\n" + readme
        if config.get(license_field):
            front["license"] = config[license_field]
        else:
            front.pop("license", None)
        (folder / "README.md").write_text(
            "---\n" + yaml.safe_dump(front, sort_keys=False) + "---" + body
        )
        (folder / "ATTRIBUTION.md").write_text(config["attribution"] + "\n")
        if name == "data":
            refresh_checksums(folder)
        else:
            links = json.loads((folder / "dataset-links.json").read_text())
            links.update(
                dataset_repo=config["dataset_repo"],
                dataset_revision=revisions["dataset"]["revision"],
                dataset_catalog_sha256=sha256(root / "data/release.json"),
            )
            dump(folder / "dataset-links.json", links)
        hashes = {
            str(p.relative_to(folder)): sha256(p)
            for p in folder.rglob("*")
            if p.is_file() and ".cache" not in p.parts and p.name != "SHA256SUMS"
        }
        (folder / "SHA256SUMS").write_text(
            "".join(f"{digest}  {path}\n" for path, digest in sorted(hashes.items()))
        )
        repo = config[repo_field]
        api.create_repo(
            repo, repo_type=kind, private=config.get("private", False), exist_ok=True
        )
        print(f"Uploading {len(hashes)} files to {kind} {repo}", flush=True)
        commit = api.upload_folder(
            repo_id=repo,
            repo_type=kind,
            folder_path=folder,
            ignore_patterns=[".cache/**", "*.partial"],
            commit_message="Publish verified conference artifacts",
        )
        revision = commit.oid
        print(f"Upload committed: {repo}@{revision}; verifying remote bytes", flush=True)
        # Verify bytes served from the immutable revision before advancing the lock.
        for number, (relative, expected) in enumerate(hashes.items(), 1):
            downloaded = hf_hub_download(
                repo, relative, repo_type=kind, revision=revision
            )
            if sha256(downloaded) != expected:
                raise ValueError(f"Uploaded file differs: {repo}/{relative}")
            if number % 100 == 0:
                print(f"Verified {number}/{len(hashes)} files in {repo}", flush=True)
        revisions[kind] = {"repo_id": repo, "revision": revision}
        dump(ROOT / "release/hosting/upload-progress.json", revisions)
    dump(ROOT / "release/hosting/download-lock.json", revisions)
    print(json.dumps(revisions, indent=2))


def finalize(output):
    data, models = output / "data", output / "models"
    catalog = ScientificCatalog(data, offline=True)
    validation = catalog.release.get("loader_validation", {})
    if validation.get("fingerprint") != validation_fingerprint(data):
        raise ValueError("Validate this exact catalog before finalizing")
    catalog.verify(assets=False)
    # Preserve generated HF configurations while refreshing the maintained card.
    front = (data / "README.md").read_text().split("---", 2)[1]
    (data / "README.md").write_text(
        "---"
        + front
        + "---\n\n"
        + (ROOT / "release/hosting/dataset-card.md").read_text()
    )
    refresh_checksums(data)
    catalog = ScientificCatalog(data, offline=True)
    hashes = dict(catalog.release["files"])
    hashes.update({r.path: r.sha256 for r in catalog.assets.itertuples()})
    hashes["release.json"] = sha256(data / "release.json")
    (data / "SHA256SUMS").write_text(
        "".join(f"{value}  {path}\n" for path, value in sorted(hashes.items()))
    )
    registry = json.loads((ROOT / "release/registry.json").read_text())
    links = {
        "dataset_revision": None,
        "dataset_catalog_sha256": hashes["release.json"],
        "experiments": {},
    }
    for identifier, model in registry["models"].items():
        if sha256(models / model["checkpoint"]) != model["sha256"]:
            raise ValueError(f"Checkpoint changed: {identifier}")
        spec = catalog.release["cohorts"].get(identifier, {})
        normalization = spec.get("metadata", {}).get("stats.json")
        links["experiments"][identifier] = {
            "cohort": identifier if spec else None,
            "normalization": normalization,
            "normalization_sha256": (
                sha256(data / normalization) if normalization else None
            ),
            "effective_cohorts": {
                split: {"path": path, "sha256": sha256(data / path)}
                for split, path in spec.get("effective_splits", {}).items()
            },
        }
    dump(models / "dataset-links.json", links)
    inventory = {
        "schema_version": 1,
        "dataset_root": "artifacts/upload/data",
        "model_root": "artifacts/upload/models",
        "assets": len(catalog.assets),
        "asset_bytes": int(catalog.assets.size_bytes.sum()),
        "samples": len(catalog.table("samples")),
        "observations": len(catalog.table("observations")),
        "source_cohorts": len(catalog.release["cohorts"]),
        "checkpoint_count": len(registry["models"]),
        "checkpoint_bytes": sum(m["size_bytes"] for m in registry["models"].values()),
        "formats": catalog.assets["format"].value_counts().to_dict(),
        "profile_bytes": {
            profile: int(
                catalog.assets.loc[
                    catalog.assets.profiles.map(lambda values: profile in values),
                    "size_bytes",
                ].sum()
            )
            for profile in ("paper-eval", "train", "storms")
        },
        "dataset_release_sha256": hashes["release.json"],
        "publication_status": "local staging only; no uploads performed",
    }
    dump(ROOT / "release/hosting/inventory.json", inventory)
    shutil.copytree(ROOT / "release", models / "release", dirs_exist_ok=True)
    (models / "README.md").write_text(
        (ROOT / "release/hosting/models-card.md").read_text()
    )
    model_hashes = {
        str(p.relative_to(models)): sha256(p)
        for p in models.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS" and ".cache" not in p.parts
    }
    (models / "SHA256SUMS").write_text(
        "".join(f"{value}  {path}\n" for path, value in sorted(model_hashes.items()))
    )
    print(json.dumps(inventory, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "verify",
            "download",
            "view",
            "storm-view",
            "finalize",
            "upload",
        ],
    )
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--publish-config", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/upload"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--revision")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["index-only", "paper-eval", "train", "storms", "all"],
        default="index-only",
    )
    parser.add_argument("--cohort")
    parser.add_argument("--split")
    parser.add_argument("--storm")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        if not args.source_root or not args.artifact_root:
            parser.error("prepare requires --source-root and --artifact-root")
        prepare(args)
    elif args.action == "finalize":
        finalize(args.output.resolve())
    elif args.action == "upload":
        if not args.publish_config:
            parser.error("upload requires --publish-config")
        upload(args)
    else:
        if not args.root:
            parser.error("--root is required")
        catalog = ScientificCatalog(
            args.root,
            repo_id=args.repo_id,
            revision=args.revision,
            offline=args.offline,
        )
        if args.action == "verify":
            print(json.dumps(catalog.verify(assets=not args.metadata_only), indent=2))
        elif args.action == "download":
            for path in catalog.release["files"]:
                if (
                    args.cohort is not None
                    or args.profile != "index-only"
                    or (
                        path.endswith(".parquet")
                        and path.startswith(
                            ("index/", "cohorts/", "effective-cohorts/")
                        )
                    )
                    or path in ("README.md", "schema.json")
                ):
                    catalog.file(path)
            print(
                "Assets verified:",
                catalog.prefetch(
                    cohort=args.cohort,
                    split=args.split,
                    storm=args.storm,
                    profile=args.profile,
                ),
            )
        elif args.action == "storm-view":
            from geo2wf.data.catalog import materialize_inference

            print(materialize_inference(catalog, args.output))
        elif args.action == "view":
            if not args.cohort:
                parser.error("view requires --cohort")
            print(catalog.dataset_root(args.cohort, args.output))


if __name__ == "__main__":
    main()
