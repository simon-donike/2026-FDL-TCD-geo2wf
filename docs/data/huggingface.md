# Indexed Hugging Face release

Run from the `conference-release` checkout after `uv sync --frozen --group dev --group docs`.
All generated binaries live under ignored `artifacts/`. Source files are copied,
never moved or rewritten. The catalog is portable; runtime views are disposable.

## Prepare and validate

```bash
uv run python scripts/release_hub.py prepare \
  --source-root /path/to/research-checkout \
  --artifact-root /path/to/conference-release-artifacts
uv run python scripts/release_hub.py verify --root artifacts/upload/data
uv run python scripts/validate_catalog_loaders.py \
  --root artifacts/upload/data \
  --source-root /path/to/research-checkout \
  --artifact-root /path/to/conference-release-artifacts
uv run python scripts/release_hub.py finalize --output artifacts/upload
```

The dataset contains Parquet and CSV indexes, original indexes, byte-identical
GeoTIFF/NetCDF/NPZ assets, normalization files, and original result reports.
The model folder contains 21 checkpoints, registry, original producer source
snapshots and training provenance. The source index includes unavailable records
explicitly. Asset identity is SHA-256; paths are relative to the dataset root.

Each cohort preserves original manifest order. `effective-cohorts/` is produced
by loader validation and records actual runtime eligibility and RI labels.
Do not interpret `eligible=null` as eligible or assume cohorts match. Original
training that included test observations remains documented in the model registry.
Original index extensions stay in JSON columns; typed shared fields support HF
browsing. Archived provenance can contain historical machine paths, but operational
asset resolution uses relative paths exclusively.

The loader uses a small generated view descriptor to read Parquet directly. CSV
views also support existing evaluators. Transform functions and model batch
contracts are unchanged. Forecast labels remain in their indexed cohort rows;
cache metadata preserves the original feature scaler and producer checkpoint.

## Inspect and download

After publication, obtain both repository IDs and full commit hashes from
`download-lock.json`. Never use a floating branch for reproducing the paper.

```python
from datasets import load_dataset
index = load_dataset("ORG/stormsense-data", "samples",
                     split="catalog", revision=DATA_COMMIT)
print(index[0])  # Metadata only; scientific pixels are separate assets.
```

The optional `datasets` library is only needed for this browsing example.
The maintained training loader uses pandas/pyarrow and Rasterio.

```bash
uv run python scripts/release_hub.py download \
  --repo-id ORG/stormsense-data --revision FULL_DATA_COMMIT \
  --root artifacts/data --profile paper-eval
uv run python scripts/release_hub.py view --root artifacts/data \
  --cohort architecture-with-era5-unet --output build/one-model
```

Profiles: `index-only`, `paper-eval`, `train`, `storms`, `all`. Explicit `--cohort`
and `--split` select asset dependencies; `--storm` restricts storm downloads.
`--offline` prevents network access. Checksums are verified even for cached assets.
Corrupt files fail explicitly; remove the reported cached file and re-download it.
Interrupted transfers resume through the HF client's download cache.

```python
from geo2wf.data.catalog import ScientificCatalog
from geo2wf.data.datasets.paired_geotiff import PairedImageDataset
catalog = ScientificCatalog("artifacts/data", offline=True)
view = catalog.dataset_root("architecture-with-era5-unet", "build/example")
sample = PairedImageDataset(view, "test", require_era5=True,
                           use_era5=True, max_era5_time_gap_hours=3.1)[0]
```

Use recorded model configs for exact evaluation/training rather than this minimal
exploration example. Download model files with `hf download ORG/stormsense-models
--revision FULL_MODEL_COMMIT --local-dir artifacts/models`.

```bash
uv run python scripts/conference_release.py evaluate-architecture \
  --artifact-root artifacts/models --catalog-root artifacts/data \
  --era5 with-era5 --output build/paper-results
uv run python scripts/conference_release.py train \
  --artifact-root artifacts/models --catalog-root artifacts/data \
  --model latent_sar_era5_max_wind --device cuda
uv run python scripts/conference_release.py storm-inference \
  --artifact-root artifacts/models --catalog-root artifacts/data \
  --era5 with-era5 --limit 1
```

Training needs `train` assets; dense storm inference needs `storms` plus the model's
paper assets and metadata. The original `--data-root` workflow remains supported.

## Publish explicitly

Copy `publish.example.json` to a local configuration and supply owner, agreed
licenses and source attribution. Authenticate using `hf auth login` or `HF_TOKEN`;
never put tokens in the tracked config.

```bash
uv run python scripts/release_hub.py upload \
  --output artifacts/upload --publish-config /path/to/publish.json
```

This is an external write: it creates/updates the configured HF repositories.
It verifies local hashes and requires loader validation, uploads data first,
records its immutable revision in the model links, uploads models, verifies
remote bytes, then writes `download-lock.json`. `upload-progress.json` records
successful repositories if a later step fails. Publish subsequent versions as
new commits; existing paper locks stay pinned. Hosted dataset-viewer rendering
must also be checked during publication; it cannot be tested before upload.

Website deployment and R2 publication are handled by the companion
the `geo2wf` application repository checkout. Missing external dashboard weights, original
configuration gaps, and numerical drift remain recorded in the conference
reproduction guide. HF publication here is limited to the scientific data and
model repositories.

The curated training release starts from the exact processed GeoTIFF exports.
Original upstream training granules are not included; rebuilding those exports
requires acquiring the source products described in the preprocessing guide.
Case-study NetCDF sources are included. The observation index distinguishes
`processed` representations from included `source` files.

## Dashboard outputs

Dashboard and explorer exports are produced in the companion
the `geo2wf` application repository checkout. This repository can materialize scientific model
and observation views for evaluation, but it does not generate website assets
or publish R2 releases.
