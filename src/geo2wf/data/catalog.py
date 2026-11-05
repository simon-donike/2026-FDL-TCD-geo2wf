"""Versioned scientific catalog: metadata-first, verified local/HF assets.

No remote dataset code is executed. Original model-facing CSV values are retained
in cohort rows; the adapter only resolves storage locations.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path, PurePosixPath

import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    part = PurePosixPath(relative)
    if part.is_absolute() or ".." in part.parts or "\\" in relative:
        raise ValueError(f"Unsafe catalog path: {relative}")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"Catalog path escapes root: {relative}")
    return path


class ScientificCatalog:
    def __init__(self, root, *, repo_id=None, revision=None, offline=False):
        self.root = Path(root).expanduser().resolve()
        self.repo_id, self.revision, self.offline = repo_id, revision, offline
        if repo_id and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("HF catalogs require a full immutable commit revision")
        self.release = json.loads(self.file("release.json").read_text())
        if self.release["schema_version"] != 1:
            raise ValueError("Unsupported scientific catalog schema")
        self.assets = self.table("assets").set_index("asset_id", drop=False)

    def file(self, relative):
        path = safe_path(self.root, relative)
        if not path.is_file():
            if self.offline or not self.repo_id:
                raise FileNotFoundError(
                    f"Not cached: {relative}; prefetch this profile first"
                )
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                self.repo_id,
                relative,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self.root,
            )
        return path

    def table(self, name):
        return pd.read_parquet(self.file(f"index/{name}.parquet"))

    def cohort(self, identifier, split):
        spec = self.release["cohorts"][identifier]
        split = "validation" if split == "val" else split
        return pd.read_parquet(self.file(spec["splits"][split]))

    def asset(self, identifier):
        row = self.assets.loc[identifier]
        path = self.file(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {identifier} ({path})")
        return path

    def prefetch(self, *, cohort=None, split=None, storm=None, profile="all"):
        ids = set()
        if cohort:
            spec = self.release["cohorts"][cohort]
            for name in ([split] if split else spec["splits"]):
                for refs in self.cohort(cohort, name)["asset_ids"]:
                    ids.update(refs)
            ids.update(spec.get("support_asset_ids", []))
        elif profile != "index-only":
            for _, row in self.assets.iterrows():
                if profile == "all" or profile in row["profiles"]:
                    if (
                        storm is None
                        or not len(row["storm_ids"])
                        or storm in row["storm_ids"]
                    ):
                        ids.add(row["asset_id"])
        for identifier in sorted(ids):
            self.asset(identifier)
        return len(ids)

    def dataset_root(self, cohort, destination, *, prefetch=True):
        """Materialize a tiny legacy view; arrays remain in the canonical catalog.

        This preserves third-party evaluators expecting root/split/manifest.csv,
        while project readers resolve the Parquet rows directly via read_manifest.
        """
        destination = Path(destination).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        if prefetch:
            self.prefetch(cohort=cohort)
        spec = self.release["cohorts"][cohort]
        (destination / "catalog-view.json").write_text(
            json.dumps(
                {
                    "root": str(self.root),
                    "cohort": cohort,
                },
                indent=2,
            )
        )
        for split, legacy in spec["legacy_splits"].items():
            directory = destination / legacy
            directory.mkdir(exist_ok=True)
            self.manifest(cohort, split).to_csv(directory / "manifest.csv", index=False)
        for name, relative in spec.get("metadata", {}).items():
            payload = self.file(relative).read_bytes()
            (destination / name).write_bytes(payload)
        return destination

    def manifest(self, cohort, split):
        records = []
        for _, entry in self.cohort(cohort, split).sort_values("ordinal").iterrows():
            row = json.loads(entry["legacy_row_json"])
            for column, asset_id in json.loads(entry["path_assets_json"]).items():
                row[column] = str(
                    safe_path(self.root, self.assets.loc[asset_id, "path"])
                )
            records.append(row)
        return pd.read_csv(
            io.StringIO(pd.DataFrame(records).to_csv(index=False)),
            keep_default_na=False,
        )

    def verify(self, *, assets=True):
        for relative, expected in self.release["files"].items():
            if sha256(self.file(relative)) != expected:
                raise ValueError(f"Catalog checksum mismatch: {relative}")
        if self.assets.index.duplicated().any():
            raise ValueError("Duplicate asset identity")
        asset_ids = set(self.assets.index)
        samples = self.table("samples")
        if samples["sample_key"].duplicated().any():
            raise ValueError("Duplicate sample identity")
        keys = set(samples["sample_key"])
        for _, sample in samples.iterrows():
            if not set(sample.asset_ids).issubset(asset_ids):
                raise ValueError("Unknown sample asset")
            for modality in ("geo", "sar", "era5", "pmw"):
                if f"has_{modality}" in sample:
                    present = pd.notna(sample[f"{modality}_asset_id"])
                    if bool(sample[f"has_{modality}"]) != present:
                        raise ValueError(f"Inconsistent {modality} availability")
        observations = self.table("observations")
        if (
            "observation_key" in observations
            and observations.observation_key.duplicated().any()
        ):
            raise ValueError("Duplicate observation representation identity")
        if "available" in observations:
            included = observations.loc[observations.available, "asset_id"]
            if not set(included).issubset(asset_ids):
                raise ValueError("Unknown observation asset")
        for name, spec in self.release["cohorts"].items():
            for split in spec["splits"]:
                rows = self.cohort(name, split)
                if list(rows.ordinal) != list(range(len(rows))):
                    raise ValueError(f"Invalid ordering: {name}/{split}")
                if not set(rows.sample_key).issubset(keys):
                    raise ValueError(f"Unknown sample: {name}/{split}")
                for refs in rows.asset_ids:
                    if not set(refs).issubset(asset_ids):
                        raise ValueError(f"Unknown asset: {name}/{split}")
        for _, row in self.assets.iterrows():
            safe_path(self.root, row["path"])
            if assets:
                self.asset(row["asset_id"])
        return {
            "assets": len(self.assets),
            "samples": len(samples),
            "cohorts": len(self.release["cohorts"]),
        }


def read_manifest(path):
    """Read legacy CSV or its catalog-backed equivalent without changing dtypes."""
    path = Path(path)
    descriptor = path.parent.parent / "catalog-view.json"
    if descriptor.is_file():
        view = json.loads(descriptor.read_text())
        catalog = ScientificCatalog(view["root"], offline=True)
        return catalog.manifest(view["cohort"], path.parent.name)
    return pd.read_csv(path, keep_default_na=False)


def configure_release(catalog, registry, work):
    """Repoint portable run configs at catalog views, without editing provenance."""
    import yaml

    work = Path(work)
    for identifier, model in registry["models"].items():
        if identifier not in catalog.release["cohorts"] or not model.get("config"):
            continue
        spec = catalog.release["cohorts"][identifier]
        view = catalog.dataset_root(
            identifier, work / "catalog-views" / identifier, prefetch=False
        )
        config = work / Path(model["config"]).relative_to("release/provenance")
        payload = yaml.safe_load(config.read_text())
        payload["data"]["root"] = str(view)
        if "stats.json" in spec["metadata"]:
            payload["data"]["stats_file"] = str(view / "stats.json")
        if spec.get("ibtracs_asset_id"):
            payload["data"]["ibtracs_file"] = str(
                catalog.asset(spec["ibtracs_asset_id"])
            )
        if "eligibility_cache_dir" in payload["data"]:
            payload["data"]["eligibility_cache_dir"] = str(view / ".eligibility")
        config.write_text(yaml.safe_dump(payload, sort_keys=False))
    export = work / Path(registry["structure_cache"]["export_config"]).relative_to(
        "release/provenance"
    )
    if export.is_file():
        payload = yaml.safe_load(export.read_text())
        view = work / "catalog-views" / registry["structure_cache"]["field_model"]
        payload["data"]["root"] = str(view)
        payload["data"]["stats_file"] = str(view / "stats.json")
        if "ibtracs_file" in payload["data"]:
            spec = catalog.release["cohorts"]["architecture-with-era5-unet"]
            payload["data"]["ibtracs_file"] = str(
                catalog.asset(spec["ibtracs_asset_id"])
            )
        export.write_text(yaml.safe_dump(payload, sort_keys=False))
    # Runtime path relocation changes YAML hashes; preserve the producer hash
    # separately and bind comparison caches to the relocated equivalent config.
    for regime, architecture in registry["architecture"].items():
        identifier = architecture["models"]["correction"]
        metadata = work / "catalog-views" / identifier / "cache-metadata.json"
        if not metadata.is_file():
            continue
        payload = json.loads(metadata.read_text())
        original = payload["unet_config"]
        original.setdefault("original_sha256", original["sha256"])
        field = registry["models"][architecture["models"]["unet"]]["config"]
        config = work / Path(field).relative_to("release/provenance")
        original.update(path=str(config), sha256=sha256(config))
        metadata.write_text(json.dumps(payload, indent=2) + "\n")
    return work / "catalog-views"


def materialize_storms(catalog, destination):
    """Portable inference view generated from the published observation index."""
    destination = Path(destination).resolve()
    directory = destination / "index-files"
    directory.mkdir(parents=True, exist_ok=True)
    observations = catalog.table("observations")
    records = []
    for _, entry in observations.iterrows():
        if not entry.get("case_study", True):
            continue
        row = json.loads(entry["metadata_json"])
        if entry["available"]:
            row["path"] = str(catalog.asset(entry["asset_id"]))
        else:
            raise FileNotFoundError(f"Storm source unavailable: {entry.observation_id}")
        records.append(row)
    pd.DataFrame(records).to_csv(directory / "observation_manifest_v6.csv", index=False)
    return destination


def materialize_inference(catalog, destination):
    """Restore exporter input paths as links to verified canonical assets."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    catalog.prefetch(profile="storms")
    mapping = pd.read_parquet(catalog.file("provenance/path-mapping.parquet"))
    for entry in mapping.itertuples():
        if not entry.original_path.startswith("inference/"):
            continue
        relative = entry.original_path.removeprefix("inference/")
        relative_path = PurePosixPath(relative)
        path = safe_path(destination, str(relative_path.parent)) / relative_path.name
        source = catalog.file(catalog.assets.loc[entry.asset_id, "path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() and path.resolve() == source.resolve():
            continue
        if path.exists():
            raise FileExistsError(
                f"Refusing to replace existing inference input: {path}"
            )
        path.symlink_to(source)
    materialize_storms(catalog, destination / "inf_data")
    return destination
