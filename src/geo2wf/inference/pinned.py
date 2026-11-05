"""Pinned, reproducible model assets and preprocessing.

This module intentionally contains only scientific checkpoint resolution and
input preparation. It has no operational scheduling, publishing, or database
dependencies, so historical experiments can be run from the science checkout.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch

from geo2wf.data.features import normalized_distance_to_center, solar_time_features
from geo2wf.data.normalization import normalize, normalization_affine_parameters

from geo2wf.observations.common import file_hash
from geo2wf.observations.satellite import BANDS


MANIFEST_PATH = Path(
    os.environ.get("GEO2WF_MODEL_MANIFEST", Path(__file__).with_name("models.json"))
).resolve()
MANIFEST = json.loads(MANIFEST_PATH.read_text())

_source_root = Path(__file__).resolve().parents[3]
# Local checkpoints live beside the source tree. An explicit root is useful for
# installed packages and archive runners, while the source-tree probe avoids
# accidentally treating a site-packages parent as the repository.
ROOT = Path(os.environ["GEO2WF_ROOT"]).expanduser().resolve() if os.environ.get(
    "GEO2WF_ROOT"
) else (_source_root if (_source_root / "pyproject.toml").is_file() else Path.cwd())


def resolve_file(name, model_root):
    """Resolve an asset from the supplied model root, then this checkout."""
    candidates = [Path(model_root).expanduser() / name, ROOT / name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing pinned model asset: {name}")


def checkpoint(role, model_root="downloads/models"):
    """Return and checksum the immutable checkpoint for *role*."""
    item = MANIFEST["models"][role]
    local_name = item.get("local_checkpoint")
    candidates = []
    if local_name:
        # local_checkpoint is relative to the science repository root, never cwd.
        candidates.append(ROOT / local_name)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        path = resolve_file(item["checkpoint"], model_root)
    if path is None:
        raise FileNotFoundError(f"Missing checkpoint for {role}: {item['checkpoint']}")
    if file_hash(path) != item["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {role}")
    return path


def prepare(array, valid, lat, lon, at, stats):
    """Prepare a GOES crop using the pinned training preprocessing."""
    values = torch.from_numpy(np.asarray(array, dtype=np.float32))
    normalized = normalize(
        values,
        "geo",
        list(BANDS),
        stats,
        normalization="robust-zscore",
        robust_clip=4.0,
    )
    mask = torch.from_numpy(valid).unsqueeze(0)
    condition = (torch.nan_to_num(normalized) * mask)[:, 32:224, 32:224]
    mask = mask[:, 32:224, 32:224]
    half = 192 * 0.027 / 2
    bounds = torch.tensor(
        [lon - half, lon + half, lat - half, lat + half], dtype=torch.float64
    )
    distance = normalized_distance_to_center(
        bounds, (192, 192), torch.tensor([lat, lon])
    )
    solar = solar_time_features(bounds, (192, 192), at)
    condition = torch.cat([condition, distance, solar], dim=0)
    offset, scale = normalization_affine_parameters(
        "sar", ["wind_speed"], stats, normalization="min-max"
    )
    return {
        "condition": condition.unsqueeze(0),
        "condition_mask": mask.unsqueeze(0),
        "target_norm_offset": offset.unsqueeze(0),
        "target_norm_scale": scale.unsqueeze(0),
    }


def training_storms(model_root="downloads/models"):
    """Read the pinned pretraining storm membership manifest."""
    item = MANIFEST["models"]["nowcast"]
    path = resolve_file(item["training_manifest"], model_root)
    if file_hash(path) != item["training_manifest_sha256"]:
        raise ValueError("Training membership manifest checksum mismatch")
    with gzip.open(path, "rt") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "storm_id" not in rows.fieldnames:
            raise ValueError("training membership unavailable")
        return {row["storm_id"] for row in rows}
