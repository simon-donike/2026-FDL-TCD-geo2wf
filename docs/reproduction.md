# Reproduce the conference release

This release preserves the two published tables and the Humberto/Kiko/Otis
case studies. Website and operational serving code are maintained in the
companion `geo2wf` application repository checkout. The exact checkpoint inventory and
source metadata are in [`release/registry.json`](../release/registry.json).

## Results and cohorts

The architecture table uses the original **212 test observations from 38
storms**, including 19 RI observations from 11 storms. It retains the six
August 20 architecture checkpoints. The eight-row latent-supervision table
uses each August 27 run's **configured validation cohort**, with the original
ERA5 eligibility filters. The joint models in the two tables are different
checkpoints. Do not substitute the later strict-test reevaluation or describe
these two tables as a single matched experiment.

All supplied paper numbers and rounding are preserved. The historical
validation record in `release/validation.json` records small CPU reevaluation
differences for two rounded values; saved original reports remain the
publication reference. The registry links
rows to original full-precision reports, resolved configurations, run manifests,
source commits, and dataset fingerprints. Compressed provenance CSVs decompress
byte-for-byte to the captured source files. Reports retain their original
absolute paths as historical evidence; runtime commands create separate
portable copies.

The case-study image-radii and MLP-radii correction heads used a frozen field
cache from `structure-cache-field`. That field's original training config has
`include_test_in_train: true`; the later cache-export config has it set to
false. Both are preserved. This dependency is separate from the paper's six
architecture checkpoints. The original configuration for the earliest
initialization checkpoint is unavailable locally; its exact weights are included.

## Install and obtain artifacts

```bash
uv sync --frozen --group dev --group docs
uv run geo2wf-evaluate conference --help
```

Extract the separately prepared `conference-release-artifacts.tar.gz` bundle.
It contains 21 exact checkpoints, saved source patches/snapshots, training
metrics, original provenance, and `SHA256SUMS`. Binary checkpoints are outside
Git. A hosted download URL will be added when the bundle is published.

```bash
cd /path/to/conference-artifacts
sha256sum -c SHA256SUMS
```

Raw observations remain external. `--data-root` is a directory containing
`geotiff/`, `IBTrACs/`, and the relevant intensity-cache data. Use the
[export workflow](data/export-geo-sar.md) to prepare data, retaining the original
split manifests and normalization statistics in the registry. `--inference-root`
points to the storm `inf_data/` directory containing
`index-files/observation_manifest_v6.csv`. Fingerprints identify the required
versions; an arbitrary fresh dataset export is not an equivalent cohort.

## Verify and regenerate saved outputs

```bash
uv run geo2wf-evaluate conference verify --artifact-root /path/to/conference-artifacts
uv run geo2wf-evaluate conference smoke --artifact-root /path/to/conference-artifacts
uv run geo2wf-evaluate conference tables
uv run geo2wf-evaluate conference figures
```

Tables are verified against full-precision source metrics. Figures and storm
metrics are regenerated from the saved native observations, with hourly means
and the original centred three-hour smoothing used only for plots. Outputs go
to `build/conference/results`; original evidence is never overwritten.

## Reevaluate checkpoints

Use the same artifact/data roots and work directory across commands:

```bash
uv run geo2wf-evaluate conference evaluate-latent \
  --artifact-root /path/to/conference-artifacts --data-root /path/to/data

# Run each command for both with-era5 and without-era5.
uv run geo2wf-evaluate conference cache --era5 with-era5 \
  --artifact-root /path/to/conference-artifacts --data-root /path/to/data
uv run geo2wf-evaluate conference evaluate-architecture --era5 with-era5 \
  --artifact-root /path/to/conference-artifacts --data-root /path/to/data
uv run geo2wf-evaluate conference storm-inference --era5 with-era5 \
  --artifact-root /path/to/conference-artifacts --data-root /path/to/data \
  --inference-root /path/to/inf_data
```

CPU is the default; use `--device cuda` for GPU runs. `storm-inference --limit 1`
is a small input-pipeline smoke check, not a scientific reevaluation. All
checkpoint selection comes from the registry. No command chooses the latest run.

## Train retained experiments

```bash
uv run geo2wf-evaluate conference train \
  --model latent_sar_era5_max_wind \
  --artifact-root /path/to/conference-artifacts --data-root /path/to/data \
  --device cuda
```

Choose any registry model with a captured training configuration. The wrapper
retains the original model, objectives, splits, seed, and initialization weights;
it relocates outputs, disables external tracking, and selects one requested
device. Generate caches first when training correction heads. Hardware and
library differences can affect newly trained weights; the bundled checkpoints
are the reference for the published values.

## Companion website and dashboard

The StormSense application, documentation deployment, explorer export, R2
publication, and browser checks belong to the companion `geo2wf` application repository
checkout. This scientific repository supplies the model checkpoints, data
catalogs, and reproducible evaluation outputs that the application consumes.
The registry retains provenance for externally supplied ConvLSTM and ViT
outputs; their unavailable training implementations are not part of this
checkout.

## Indexed Hugging Face data

The [indexed dataset workflow](data/huggingface.md) supports portable downloads, original indexes, exact experiment cohorts, and training from the published scientific assets.
