# M.Tech Project Status Summary

## Overall

The project is a staged multimodal drug-discovery pipeline under `multimodal_datapipeline/`. It is organized for molecule, protein, image, paired-modality, and full multimodal experiments.

## Data Pipeline Done

- ChEMBL multi-target activity data downloaded and combined.
- 12 protein targets selected from UniProt/ChEMBL.
- AlphaFold PDB structures downloaded for all 12 targets.
- BBBC021 metadata and Week 1 image ZIPs downloaded/extracted.
- Dataset manifest exists at `multimodal_datapipeline/dataset_pipeline_output/manifest.json`.

Current raw/intermediate counts:

- ChEMBL multitarget activity rows: `96,188`
- Curated molecule-target rows: `46,915`
- Unique curated SMILES: `42,033`
- AlphaFold structures: `12`
- BBBC021 image metadata rows: `13,200`

## Processed Tables Done

- Molecule-only curated ChEMBL table: `46,915` rows.
- Protein-only table: `12` targets, all with sequences and activity labels.
- Image-only BBBC021 table: `516` usable rows, `8` compounds, `5` MoA classes.
- Molecule + protein table: `46,915` rows, `12` targets, `42,033` molecules.
- Molecule + image table: `300` rows, `6` compounds, `3` MoA classes.
- Protein + image is blocked because BBBC021 rows do not contain target/protein annotations.
- Full molecule + protein + image is blocked because there is `0` exact SMILES overlap between ChEMBL molecule-protein data and BBBC021 molecule-image data.

## Model Code Done

Implemented model components:

- `src/multimodal_datapipeline/models/molecule_encoder.py`: RDKit Morgan fingerprint + MLP.
- `src/multimodal_datapipeline/models/protein_encoder.py`: ESM-2 encoder.
- `src/multimodal_datapipeline/models/image_encoder.py`: DINOv2 image encoder.
- `src/multimodal_datapipeline/models/fusion.py`: concatenation fusion for 2 or 3 modalities.

## Baselines Status

- Baseline 1, molecule-only: implemented and run.
  - Best saved test result: accuracy `0.813`, F1 `0.897`, ROC-AUC `0.741`.
- Baseline 2, protein-only: implemented and run recently.
  - Uses ESM-2 frozen embeddings + MLP regression on `active_fraction`.
  - Test RMSE `0.0659`, MAE `0.0596`, but R2 is very poor because there are only `12` target-level rows.
  - This result folder is currently untracked in git.
- Baseline 3, image-only: data curation and validation exist, but training is not implemented yet.
  - `train.py` intentionally exits with "image model training is not implemented yet."
- Baseline 4, molecule + protein: data table exists, but training is only a guarded placeholder.
- Baseline 5, molecule + image: data table exists, but training is only a guarded placeholder.
- Baseline 6, protein + image: blocked by missing protein-image alignment.
- Baseline 7, full multimodal: blocked by no molecule overlap between ChEMBL and BBBC021.

## Experiments Saved

Existing experiment outputs:

- `baseline_1_smoke_test`
- `molecule_only_baseline`
- `molecule_only_baseline_wrapper_test`
- `molecule_encoder_validation`
- `baseline_2_protein_only`

Git status from `Mtech_project`: only `multimodal_datapipeline/experiments/baseline_2_protein_only/` is untracked.

## Main Remaining Work

The next concrete step is Baseline 3 training: load the 3-channel BBBC021 images, extract DINOv2 embeddings, and train MoA classification.

After that, implement Baseline 4 and Baseline 5 fusion training. For Baseline 6 and Baseline 7, the real blocker is biological/data alignment, not code structure.
