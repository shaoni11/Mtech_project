# Experiments Conducted

## 1. Molecule Encoder Validation

- Folder: `multimodal_datapipeline/experiments/molecule_encoder_validation/`
- Checked `52,811` unique SMILES.
- All SMILES passed validation.
- Encoder smoke test passed with embedding shape `[16, 256]`.

## 2. Baseline 1 Smoke Test: Molecule-Only

- Folder: `multimodal_datapipeline/experiments/baseline_1_smoke_test/`
- Task: molecule-only activity classification.
- Model: hashed SMILES character n-grams + SGD logistic regression.
- Test ROC-AUC: `0.702`
- Test F1: `0.895`
- This was a short 1-epoch sanity run.

## 3. Baseline 1 Full Molecule-Only Run

- Folder: `multimodal_datapipeline/experiments/molecule_only_baseline/`
- Task: molecule-only activity classification on curated ChEMBL data.
- Input: `42,033` unique SMILES.
- Class balance: `34,024` active, `8,009` inactive.
- Best saved test metrics:
  - Accuracy: `0.813`
  - F1: `0.897`
  - ROC-AUC: `0.741`

## 4. Baseline 1 Wrapper Test

- Folder: `multimodal_datapipeline/experiments/molecule_only_baseline_wrapper_test/`
- Same molecule-only model, run through the wrapper script.
- Test ROC-AUC: `0.702`
- Test F1: `0.895`
- Mainly confirms `run.py` works.

## 5. Baseline 2 Protein-Only Run

- Folder: `multimodal_datapipeline/experiments/baseline_2_protein_only/`
- Task: protein-only regression.
- Model: frozen ESM-2 encoder + MLP head.
- Input: `12` protein targets.
- Target: `active_fraction`.
- Test metrics:
  - RMSE: `0.0659`
  - MAE: `0.0596`
  - R2: `-44.42`
- Important caveat: this experiment is very small, only `12` target-level rows, so the R2 is not reliable.

## Not Yet Conducted

No actual training experiments have been conducted yet for:

- Baseline 3: image-only
- Baseline 4: molecule + protein
- Baseline 5: molecule + image
- Baseline 6: protein + image
- Baseline 7: molecule + protein + image

For those, data preparation exists for Baselines 3, 4, and 5, but training is still pending.
