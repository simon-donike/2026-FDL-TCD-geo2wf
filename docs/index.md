# geo2wf scientific workflows

This documentation covers the reproducible scientific package for tropical
cyclone wind-field reconstruction and intensity estimation from geostationary
imagery. The website and operational serving application are maintained in the
companion application repository.

## Start here

- [Installation and scientific dependencies](getting-started/installation.md)
- [First experiment](getting-started/first-experiment.md)
- [Reproduce the published results](reproduction.md)
- [Dataset and Hugging Face access](data/index.md)
- [Indexed release workflow](data/huggingface.md)
- [Published cohorts](data/published-cohorts.md)
- [Models and checkpoints](models/index.md)
- [Training](experiments/training.md) and [evaluation](experiments/evaluation.md)
- [Scientific command reference](reference/commands.md)

## Scientific scope

Field models reconstruct the observation time. A separate scalar model predicts
six-hour intensity change. SAR provides training supervision for spatial wind
fields; inference uses GEO imagery and deterministic context, with ERA5 as an
optional input. The paper's architecture and latent-ablation cohorts are kept
separate in the release registry and reproduction guide.

The pinned data and model repositories are available on
[Hugging Face](https://huggingface.co/datasets/simon-donike/geo2wf-data) and
[Hugging Face](https://huggingface.co/simon-donike/geo2wf-models). Scientific
assets and checkpoint binaries are downloaded separately from Git.

See the [paper reasoning](concepts/problem.md), [results](results.md), and
[code map](reference/index.md) for context and repository boundaries.
