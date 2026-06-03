# Multimodal Data Pipeline

This project is organized to support a staged M.Tech workflow:
- dataset ingestion and curation
- single-modal baselines
- multimodal fusion experiments
- reporting and thesis artifacts

## Structure

`src/multimodal_datapipeline/`
- Core Python package.

`src/multimodal_datapipeline/pipelines/`
- End-to-end pipeline entry modules.

`src/multimodal_datapipeline/data/`
- Dataset loading, curation, and manifest logic.

`src/multimodal_datapipeline/models/`
- Ligand, protein, image, and fusion model code.

`src/multimodal_datapipeline/training/`
- Training loops, evaluation, and experiment helpers.

`configs/`
- Dataset, model, and experiment configuration files.

`scripts/`
- Runnable wrappers for common pipeline tasks.

`data/raw/`
- Raw downloaded files.

`data/interim/`
- Partially cleaned files.

`data/processed/`
- Final trainable tables and features.

`data/external/`
- Third-party datasets added manually.

`experiments/`
- Saved experiment outputs, metrics, and logs.

`reports/`
- Figures, tables, and thesis-ready exports.

`notebooks/`
- Exploration notebooks only. Keep core logic in `src/`.

## Current entrypoint

Run the dataset ingestion pipeline with:

```bash
PYTHONPATH=src python datapipeline.py --help
```
