# geo2wf-models: StormSense models and pretrained checkpoints

21 original scientific PyTorch Lightning checkpoints, exact configurations, provenance, evaluation reports and pretrained initialization dependencies for the geo2wf conference release. These are not Transformers models and do not provide a generic Transformers inference pipeline.

- **Data:** [simon-donike/geo2wf-data](https://huggingface.co/datasets/simon-donike/geo2wf-data)
- **Code:** [simon-donike/geo2wf](https://github.com/simon-donike/geo2wf)
- **Documentation and StormSense:** [tcd.hyperalis.com](https://tcd.hyperalis.com/)
- **Immutable data revision:** `dataset-links.json` links every experiment to its dataset cohort, normalization and effective split hashes.

## Checkpoint inventory

| Group | Count | Purpose |
|---|---:|---|
| Architecture comparisons | 6 | U-Net, correction and joint models, with/without ERA5 |
| Latent ablations | 8 | SAR, ERA5 and wind/radii supervision variants |
| Additional correction heads | 2 | Image and MLP radii models |
| Structure-cache field | 1 | Field producer used by case studies and dashboard |
| Dashboard heads | 2 | Forecast MLP and correction model |
| Pretrained training initializers | 2 | Exact upstream initialization dependencies |

All checkpoint paths, SHA-256 hashes, source revisions, original configurations and output relationships are recorded in `release/registry.json`. Checkpoints occupy 1,268,706,234 bytes. `run-provenance/` retains original training/source evidence. `SHA256SUMS` covers all packaged files.

## Download and run

Pin the model repository to an immutable commit. Obtain the corresponding dataset commit from `dataset-links.json`.

```bash
hf download simon-donike/geo2wf-models --revision FULL_MODEL_COMMIT --local-dir artifacts/models
```

`code/conference-source.tar.gz` contains the matching release implementation and dependency lock, including the catalog loaders and reproduction scripts. Extract into a new working directory; the archive contains a `geo2wf/` top-level folder. `code/source-manifest.json` records its base Git commit and exact included-file hashes; it includes release preparation changes beyond that commit.

From the extracted source checkout:

```bash
uv sync --frozen --group dev --group docs
uv run python scripts/release_hub.py download   --repo-id simon-donike/geo2wf-data --revision FULL_DATA_COMMIT   --root artifacts/data --profile paper-eval
uv run python scripts/conference_release.py evaluate-architecture   --artifact-root /absolute/path/to/artifacts/models --catalog-root artifacts/data   --era5 with-era5 --output build/paper-results
```

To continue training, prefetch the `train` profile and use the original experiment configuration:

```bash
uv run python scripts/conference_release.py train   --artifact-root /absolute/path/to/artifacts/models --catalog-root artifacts/data   --model latent_sar_era5_max_wind --device cuda
```

See bundled `release/hosting/README.md`, `docs/reproduction.md` and `docs/data/huggingface.md` for individual sample loading, offline use, ablation evaluation and storm reproduction. Models retain their original checkpoint-loading compatibility; class/configuration selection is explicit rather than inferred from the latest run.

## Validation and limitations

All 21 checkpoints were loaded and exercised. The migrated loaders match legacy tensors, masks, labels and metadata, with seeded augmentation tested separately. All 36 freshly evaluated architecture/latent MAEs match the established CPU reproduction baseline. Original published table values remain unchanged; two CPU results cross the final published rounding digit. Complete saved storm exports match the original workflow; fresh neural inference was checked on one observation per storm in both ERA5 regimes, not on every dense observation.

Architecture test and latent validation cohorts differ. Historical auxiliary field training included test observations; original membership and producer provenance are preserved. The earliest pretrained initializer has no recovered original training configuration. External dashboard ViT checkpoint identity and ConvLSTM weights are unavailable; their available exported outputs are in the dataset, not represented as included checkpoints.

These research models are intended for reproducibility and further scientific work; validation here is not an operational forecasting assessment.

## Attribution and licensing

Release maintainer: [Simon Donike](https://huggingface.co/simon-donike). Source and experiment attribution are preserved in the registry and provenance. See `ATTRIBUTION.md`. License metadata reflects the maintainer's explicit selection; if absent, no new reuse license is asserted by this card.
