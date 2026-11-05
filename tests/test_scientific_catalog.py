import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from geo2wf.data.catalog import ScientificCatalog, read_manifest, safe_path, sha256
from scripts.release_hub import Builder


@pytest.fixture
def catalog(tmp_path):
    source = tmp_path / "source"
    data = source / "data/demo"
    (data / "train").mkdir(parents=True)
    (data / "train/a.npz").write_bytes(b"original scientific asset")
    pd.DataFrame(
        [
            {
                "sample_id": "a",
                "storm_id": "AL012020",
                "split": "train",
                "field_path": "train/a.npz",
                "value": 1.125,
                "missing": "",
            }
        ]
    ).to_csv(data / "train/manifest.csv", index=False)
    b = Builder(source, tmp_path / "catalog")
    b.cohort("experiment", data, [data / "train/manifest.csv"])
    b.observations = [{"observation_id": "a", "storm_id": "AL012020"}]
    b.finish({"source_commit": "test"})
    return ScientificCatalog(b.output, offline=True), data


def test_manifest_values_and_paths(catalog, tmp_path):
    c, original = catalog
    view = c.dataset_root("experiment", tmp_path / "view")
    old = read_manifest(original / "train/manifest.csv")
    new = read_manifest(view / "train/manifest.csv")
    pd.testing.assert_frame_equal(
        old.drop(columns="field_path"), new.drop(columns="field_path")
    )
    assert Path(new.iloc[0].field_path).read_bytes() == b"original scientific asset"
    assert c.verify() == {"assets": 1, "samples": 1, "cohorts": 1}


def test_corrupt_asset_rejected(catalog):
    c, _ = catalog
    asset = c.assets.iloc[0]
    (c.root / asset.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        c.prefetch(profile="train")


def test_metadata_only_does_not_require_arrays(catalog):
    c, _ = catalog
    (c.root / c.assets.iloc[0].path).unlink()
    assert c.prefetch(profile="index-only") == 0
    c.verify(assets=False)
    with pytest.raises(FileNotFoundError, match="prefetch"):
        c.prefetch(cohort="experiment")


def test_revision_must_be_immutable(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        ScientificCatalog(tmp_path, repo_id="org/data", revision="main")


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../../escape", "a\\b"])
def test_reject_unsafe_paths(tmp_path, path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, path)


def test_deduplicate_by_bytes_and_union_profiles(catalog):
    c, original = catalog
    b = Builder(original.parents[1], c.root.parent / "second")
    first = b.asset(original / "train/a.npz", "assets/a.npz", profiles=["train"])
    second = b.asset(original / "train/a.npz", "assets/b.npz", profiles=["storms"])
    assert first == second
    assert len(b.assets) == 1
    assert b.assets[first]["profiles"] == ["storms", "train"]


def test_selective_remote_download_then_offline(catalog, tmp_path, monkeypatch):
    import shutil
    import huggingface_hub

    source, _ = catalog
    calls = []

    def download(repo_id, filename, *, repo_type, revision, local_dir):
        assert repo_type == "dataset" and revision == "a" * 40
        calls.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source.root / filename, target)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    root = tmp_path / "download"
    remote = ScientificCatalog(root, repo_id="org/data", revision="a" * 40)
    assert calls == ["release.json", "index/assets.parquet"]
    assert remote.prefetch(cohort="experiment") == 1
    first_calls = len(calls)
    assert remote.prefetch(cohort="experiment") == 1
    assert len(calls) == first_calls
    offline = ScientificCatalog(root, offline=True)
    assert offline.prefetch(cohort="experiment") == 1


def test_interrupted_transfer_does_not_become_valid_asset(
    catalog, tmp_path, monkeypatch
):
    import shutil
    import huggingface_hub

    source, _ = catalog
    root = tmp_path / "download"

    def failed(repo_id, filename, *, repo_type, revision, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if filename.startswith("assets/"):
            target.with_suffix(".partial").write_bytes(b"incomplete")
            raise ConnectionError("interrupted")
        shutil.copyfile(source.root / filename, target)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", failed)
    remote = ScientificCatalog(root, repo_id="org/data", revision="a" * 40)
    with pytest.raises(ConnectionError):
        remote.prefetch(cohort="experiment")
    with pytest.raises(FileNotFoundError):
        ScientificCatalog(root, offline=True).prefetch(cohort="experiment")


def test_exporter_view_restores_deduplicated_paths_and_is_repeatable(
    catalog, tmp_path, monkeypatch
):
    import geo2wf.data.catalog as module

    c, _ = catalog
    entry = c.assets.iloc[0]
    pd.DataFrame(
        [
            {
                "original_path": "inference/inf_vit/STORM/one.pt",
                "asset_id": entry.asset_id,
            },
            {
                "original_path": "inference/inf_vit/STORM/two.pt",
                "asset_id": entry.asset_id,
            },
        ]
    ).to_parquet(c.root / "provenance/path-mapping.parquet", index=False)
    monkeypatch.setattr(module, "materialize_storms", lambda catalog, output: output)
    root = module.materialize_inference(c, tmp_path / "exporter")
    first = root / "inf_vit/STORM/one.pt"
    second = root / "inf_vit/STORM/two.pt"
    assert first.is_symlink() and second.is_symlink()
    assert first.resolve() == second.resolve() == (c.root / entry.path)
    module.materialize_inference(c, root)
    assert first.read_bytes() == b"original scientific asset"
