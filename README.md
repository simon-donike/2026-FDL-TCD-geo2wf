# geo2wf scientific workflows

This repository contains the scientific source for geo2wf: data contracts and
loaders, wind-field and intensity models, training, evaluation, and the
reproduction workflows behind the published results. Website, dashboard, and
operational serving code stay in the companion application repository.
The original `geo2wf` repository is unchanged; this is an independent scientific
copy prepared for handoff.

The package distribution remains named `geo2wf` and is version `0.2.0`. The
canonical import path is `src/geo2wf`; reproducible workflows are collected
under `src/geo2wf/workflows` and exposed through small script wrappers in
`scripts/`. Scientific assets and checkpoints are downloaded separately. The extraction inventory
and source fingerprints are recorded in [`release/extraction.json`](release/extraction.json).

## Install

Python 3.10 or 3.11 is supported:

```bash
uv sync --frozen --group dev --group docs
```

CPU and CUDA PyTorch extras are preserved:

```bash
uv sync --extra cpu
uv sync --extra cuda
```

The source distribution includes package configuration and the pinned
`geo2wf.inference/models.json` catalog. Select immutable data and model IDs
from `release/registry.json` and [`docs/data/huggingface.md`](docs/data/huggingface.md).

Download the pinned release metadata first:

```bash
uv run python scripts/download_artifacts.py metadata
uv run python scripts/download_artifacts.py all --dry-run
```

## Reproduce the published results

The release preserves the architecture and latent supervision tables, cohorts,
full precision reports, resolved configurations, and checkpoint provenance.
Start with [`docs/reproduction.md`](docs/reproduction.md):

```bash
uv run geo2wf-evaluate conference verify --artifact-root /path/to/artifacts
uv run geo2wf-evaluate conference smoke --artifact-root /path/to/artifacts
uv run geo2wf-evaluate conference tables
uv run geo2wf-evaluate conference figures
```

For reevaluation, provide pinned data and checkpoint roots:

```bash
uv run geo2wf-evaluate conference evaluate-architecture \
  --artifact-root /path/to/artifacts --data-root /path/to/data \
  --era5 with-era5 --output build/paper-results
```

The paper tables retain their original cohorts. The architecture table uses
212 test observations from 38 storms; the latent table uses each run's
configured validation cohort. [`docs/data/published-cohorts.md`](docs/data/published-cohorts.md)
records published split memberships and counts.

## Train and evaluate

For a new experiment, use a local processed GEO/SAR export and an explicit
configuration:

```bash
WANDB_DISABLED=true uv run geo2wf-train \
  data=geo_sar_common10_era5 \
  model=deterministic_residual \
  data.root=/path/to/geo_sar \
  data.stats_file=/path/to/geo_sar/stats.json
```

The complete configuration and data contracts are documented in
[`docs/experiments/training.md`](docs/experiments/training.md) and
[`docs/data/dataset-contract.md`](docs/data/dataset-contract.md). Use the
catalog loader for portable published assets:

```python
from geo2wf.data.catalog import ScientificCatalog

catalog = ScientificCatalog("artifacts/data", offline=True)
samples = catalog.table("samples")       # metadata only; no pixel download
print(samples.head())
# Materialize a legacy train/validation/test view only after prefetching a cohort:
# view = catalog.dataset_root("architecture-with-era5-unet", "build/example")
```

Run data-free checks locally with:

```bash
uv run python -m pytest
```

## Provenance and handoff

This extraction was made from parent revision
`4759bd0d5bd5915730ee64676c3f27e510f3ed0e` and incorporates the conference
release at `d8b01875f4ba6de6aeddae67aaa6882aa5c0ebb7`. The
conference checkout contained uncommitted scientific documentation, catalog,
release-hosting, and validation work; those files are included here as the
handoff state. `release/` records the registry, validation, provenance
snapshots, publication tables, and hosting metadata. Historical material under
`archived/` is retained as clearly noncanonical reference code.

The upstream checkout did not contain a license file; this extraction preserves
that state.
