# Scientific code map

This checkout contains the reproducible research package. Run commands from
the scientific repository root; the website and operational application are
maintained in the companion application repository.

| Location | Responsibility |
|---|---|
| `src/geo2wf/data/` | Raster/cache loading, contracts, normalization, features, sampling, and augmentation |
| `src/geo2wf/models/` | Field, joint/encoder-only, correction, and forecast models |
| `src/geo2wf/config/` | Hydra composition, compatibility checks, and configuration loading |
| `src/geo2wf/training.py` | Lightning training loops and experiment setup |
| `src/geo2wf/metrics/`, `evaluation/` | Physical errors, storm structure, and evaluation adapters |
| `src/geo2wf/visualization/`, `tracking/` | Figures, CSV/W&B logging, and run provenance |
| `src/geo2wf/workflows/` | Scientific reproduction and publication workflows |
| `scripts/` | Thin wrappers for data preparation, inference, evaluation, and release checks |
| `configs/` | Grouped training presets and export configuration |
| `tests/` | Data/model contracts, numerical behavior, integration, and catalog checks |
| `release/` | Checkpoint registry, paper tables, validation records, and provenance |
| `archived/` | Historical research workflows retained as noncanonical reference |

The root `train.py` and compatibility modules under `src/` preserve existing
imports and checkpoints. New model and loader work belongs under `src/geo2wf/`.
Data and model configs name their `_target_`; training composes and validates
them without a central architecture registry.

Large scientific inputs and model releases belong in the linked
[Hugging Face repositories](../data/index.md). Use [commands](commands.md) for
entry points, [training](../experiments/training.md) for configuration, and
[troubleshooting](troubleshooting.md) for common failures.
