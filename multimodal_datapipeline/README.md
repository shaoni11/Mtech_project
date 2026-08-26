# Multimodal Data Pipeline

This project is organized to support a staged M.Tech workflow:
- dataset ingestion and curation
- single-modal baselines
- multimodal fusion experiments
- reporting and thesis artifacts

## Structure

`configs/`
- Dataset, model, and experiment configuration files.

`package/multimodal_datapipeline/`
- Importable Python package. Put reusable model, pipeline, training, and utility code here.

`workflows/`
- Runnable research workflows. These call package code to curate data, validate inputs, train baselines, or download datasets.

`data/raw/`
- Raw downloaded files.

`data/acquire_data.py`
- Unified data acquisition script for ChEMBL, AlphaFold, BBBC021, optional scraping, and cytoskeleton ChEMBL acquisition.

`data/pipelines/`
- Data acquisition pipeline orchestration modules.

`data/sources/`
- Source-specific acquisition helpers for ChEMBL, AlphaFold, BBBC021, and optional HTML scraping.

`data/interim/`
- Partially cleaned files.

`data/processed/`
- Final trainable tables and features.

`data/external/`
- Third-party datasets added manually.

`results/`
- Saved experiment outputs, metrics, and logs.

`reports/`
- Figures, tables, and thesis-ready exports.

`exploration/`
- Scratch notebooks and one-off analysis only. Keep reusable logic in `package/`.

## Package Subfolders

`package/multimodal_datapipeline/models/`
- Ligand, protein, image, and fusion model components.

`package/multimodal_datapipeline/utils/`
- Shared IO, filesystem, and baseline launcher helpers.

## Current entrypoint

For local development, install the project in editable mode:

```bash
python3 -m pip install -e .
```

Run the dataset ingestion pipeline with either command:

```bash
multimodal-datapipeline --help
python3 data/acquire_data.py --help
```

Run the cytoskeleton ChEMBL acquisition with:

```bash
python3 data/acquire_data.py cytoskeleton-chembl
```

Run baseline training through console entrypoints:

```bash
mmdp-baseline-1 --help
mmdp-baseline-3 --help
```

## Pants

This project includes Pants config for dependency locking and Python target ownership.

Install the Pants launcher first if `pants` is not available on your shell path. Use the official launcher install method for your OS; do not install `pantsbuild.pants` with pip, because that PyPI package only exposes old Pants v1 releases.

On macOS ARM:

```bash
brew install pantsbuild/tap/pants
```

Generate or refresh the Python lockfile from the project root:

```bash
cd multimodal_datapipeline
pants generate-lockfiles
```

Useful Pants commands:

```bash
pants list ::
pants dependencies data/acquire_data.py
pants fmt ::
```

Pants writes its generated dependency lockfile to:

```text
3rdparty/python/default.lock
```
