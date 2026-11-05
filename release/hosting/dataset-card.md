# geo2wf-data: StormSense scientific dataset

Scientific inputs, complete indexes and reference outputs for tropical-cyclone wind-field reconstruction, intensity estimation and storm case studies.

- **Models and pretrained initializers:** [simon-donike/geo2wf-models](https://huggingface.co/simon-donike/geo2wf-models)
- **Code:** [geo2wf](https://github.com/simon-donike/geo2wf). A matching conference source snapshot is also distributed in the model repository under `code/`.
- **Documentation and StormSense:** [tcd.hyperalis.com](https://tcd.hyperalis.com/)
- **Exact data contract:** `schema.json`; release inventory and checksums: `release.json` and `SHA256SUMS`.

## Contents

The release includes 96,109 catalog samples, 14,471 observation representations, 22 source cohorts and 18,940 native assets (32,884,973,800 bytes). These are heterogeneous catalogs, not 96,109 independent satellite images. Assets include original GeoTIFF imagery, NetCDF storm sources, derived NPZ fields, forecast caches, labels and reference outputs. Optional modalities are absent when unavailable.

| Directory | Purpose |
|---|---|
| `index/` | Samples, observations, assets and storms in typed Parquet and downloadable CSV |
| `cohorts/` | Original ordered experiment manifests, without duplicating assets |
| `effective-cohorts/` | Validated loader eligibility, RI labels and historical training membership |
| `assets/` | Scientific files grouped by processing version, storm and producer |
| `normalization/` | Original fixed statistics; never recompute automatically |
| `labels/` | Track and intensity references |
| `reference-results/` | Paper, case-study and dashboard outputs |
| `provenance/` | Original indexes, processing metadata and reversible path mapping |

Channel order, units, shapes, CRS, source/producer identifiers, byte sizes and SHA-256 digests are recorded in the asset index where applicable. Shared timestamps, lists, booleans and nulls are typed; original fields remain in documented JSON extensions. Asset identity is based on exact bytes. Processing versions retain distinct sample identities.

## Browse without downloading imagery

```python
from datasets import load_dataset
# Replace with an immutable commit from the model repo's dataset-links.json.
revision = "FULL_DATA_COMMIT"
samples = load_dataset("simon-donike/geo2wf-data", "samples",
                       split="catalog", revision=revision)
print(samples[0])
```

The `samples`, `observations`, `assets` and `storms` configurations expose catalog metadata. Named experiment configurations expose only available splits. Inventory configurations describe source exports, not published evaluation cohorts.

## Download and reproduce

Use the matching source snapshot from `geo2wf-models/code/` and follow its `release/hosting/README.md`. Install with `uv sync --frozen --group dev --group docs`.

```bash
uv run python scripts/release_hub.py download   --repo-id simon-donike/geo2wf-data --revision FULL_DATA_COMMIT   --root artifacts/data --profile paper-eval
```

Profiles are `index-only`, `paper-eval`, `train`, `storms` and `all`. Select a cohort/split or storm to restrict imagery downloads. Prefetch before training; local loading supports `--offline` and verifies hashes. The shared catalog reader retains original Rasterio decoding, normalization, masks, cropping and augmentation contracts. Full evaluation and training commands are in the model card and bundled guide.

## Scientific scope and limitations

The original architecture **test** results and latent-ablation **validation** results use different cohorts. There is no universal split. Historical training/test overlap for the auxiliary field model is recorded explicitly; do not interpret this as a new strictly held-out benchmark. Later `table_2` reevaluations are outside this release.

The exact processed training GeoTIFFs are included. Upstream raw training granules are not bundled; rebuilding those exports requires acquiring the source products. Source NetCDFs for Humberto, Kiko and Otis are included. Observation representations distinguish processed products from included raw sources.

Original paper tables and full-precision reports are retained. Fresh CPU evaluation has two documented last-digit rounding crossings; see model provenance and validation reports. Missing external ViT/ConvLSTM dashboard weights remain documented; their available exported results are included.

## Attribution and licensing

Release maintainer: [Simon Donike](https://huggingface.co/simon-donike). Original source/producer metadata is preserved alongside assets, including IBTrACS and ERA5 references. See `ATTRIBUTION.md`; redistribution does not replace the attribution or terms of upstream scientific products. License metadata reflects the maintainer's explicit selection; if absent, no new reuse license is asserted by this card.
