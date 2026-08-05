# M.Tech Project Status Summary

Last updated: 2026-07-26

## Overall

`multimodal_datapipeline/` is a staged multimodal drug-discovery data and baseline pipeline. It currently supports:

- dataset ingestion from ChEMBL, AlphaFold, and BBBC021
- processed single-modality and paired-modality tables
- reusable molecule, protein, image, and fusion model components
- runnable baselines for molecule-only, protein-only, and image-only experiments
- guarded placeholders for molecule-protein, molecule-image, protein-image, and full multimodal fusion training

The pipeline is usable for staged experiments, but the full molecule + protein + image model is still blocked by dataset alignment.

## Data Ingestion Status

Dataset ingestion entrypoint:

```bash
cd Mtech_project/multimodal_datapipeline
PYTHONPATH=src python datapipeline.py --help
```

Implemented ingestion modules:

- `src/multimodal_datapipeline/data/chembl.py`: ChEMBL activity and target lookup API helpers.
- `src/multimodal_datapipeline/data/alphafold.py`: AlphaFold metadata and PDB download helpers.
- `src/multimodal_datapipeline/data/bbbc021.py`: BBBC021 metadata/image ZIP download and extraction helpers.
- `src/multimodal_datapipeline/data/scrape.py`: optional HTML table scraping helper.
- `src/multimodal_datapipeline/pipelines/dataset_pipeline.py`: master CLI orchestration.

Current ingestion outputs:

| Dataset | Status | Output |
|---|---|---|
| ChEMBL multi-target activities | Done | `dataset_pipeline_output/chembl/activities_multitarget.csv` |
| ChEMBL per-target activities | Done | `dataset_pipeline_output/chembl/activities_by_target/` |
| ChEMBL target mapping | Done | `dataset_pipeline_output/chembl/target_mapping.csv` |
| AlphaFold structures | Done | `dataset_pipeline_output/alphafold/structures/` |
| AlphaFold metadata | Done | `dataset_pipeline_output/alphafold/metadata.csv` |
| BBBC021 metadata | Done | `dataset_pipeline_output/bbbc021/*.csv` |
| BBBC021 Week 1 image ZIPs | Done | `dataset_pipeline_output/bbbc021/zips/` |
| BBBC021 extracted images | Present | `dataset_pipeline_output/bbbc021/images/` |
| Manifest | Done | `dataset_pipeline_output/manifest.json` |

Current ingestion counts:

| Item | Count |
|---|---:|
| ChEMBL target count | 12 |
| ChEMBL raw multi-target activity rows | 96,188 |
| ChEMBL target-mapping rows | 112 |
| Primary single-protein targets | 12 |
| AlphaFold PDB structures | 12 |
| BBBC021 download-manifest rows | 9 |
| BBBC021 Week 1 ZIPs requested | 6 |

Selected protein targets:

```text
P00533, P31749, P31751, Q9Y243, P15056, P28482,
P27361, P24941, P35968, P12931, P00519, P42345
```

## Processed Data Status

Processed tables are under:

```text
multimodal_datapipeline/data/processed/
```

| Table | Rows | Status |
|---|---:|---|
| `chembl_molecule_curated.csv` | 46,915 | Done |
| `baseline_2_protein_only.csv` | 12 | Done |
| `baseline_3_image_only.csv` | 516 | Done |
| `baseline_4_molecule_protein.csv` | 46,915 | Done |
| `baseline_5_molecule_image.csv` | 300 | Done |
| `baseline_6_protein_image.csv` | 0 | Blocked |
| `baseline_7_molecule_protein_image.csv` | 0 | Blocked |

### Curated ChEMBL Molecule Table

Input:

```text
dataset_pipeline_output/chembl/activities_multitarget.csv
```

Output:

```text
data/processed/chembl_molecule_curated.csv
```

Curation logic:

- keeps exact `IC50` rows in `nM`
- requires usable `pchembl_value`
- standardizes molecules with RDKit
- keeps largest fragment when salts/fragments exist
- removes metals and non-organic records
- applies basic drug-like filters for heavy atoms, molecular weight, logP, HBD, and HBA
- aggregates repeated molecule-target measurements
- labels activity using `pChEMBL >= 6.0`

Key counts:

| Item | Count |
|---|---:|
| Raw ChEMBL rows | 96,188 |
| Kept activity rows before aggregation | 72,023 |
| Dropped by activity-quality filter | 23,018 |
| Aggregated molecule-target rows | 46,915 |
| Unique curated SMILES | 42,033 |
| Unique targets | 12 |
| Active rows | 37,261 |
| Inactive rows | 9,654 |

### Baseline 2 Protein-Only Table

Output:

```text
data/processed/baseline_2_protein_only.csv
```

Columns include:

```text
target_chembl_id, uniprot_id, pref_name, organism, alphafold_pdb_path,
protein_sequence, sequence_length, n_molecule_target_rows, active_fraction, label
```

Status:

- 12 target rows
- 12 rows with extracted protein sequence
- 12 rows with activity labels
- AlphaFold PDB paths retained for future structure/contact-map experiments

Important caveat:

All target-level binary labels are active, so the runnable protein-only baseline uses `active_fraction` regression rather than binary classification.

### Baseline 3 Image-Only Table

Output:

```text
data/processed/baseline_3_image_only.csv
```

Columns include:

```text
compound, smiles, concentration, moa, plate, well, replicate,
dapi_path, tubulin_path, actin_path
```

Status:

- 516 usable 3-channel microscopy rows
- 8 unique compounds
- 5 MoA classes
- 3,332 BBBC021 rows skipped because image channel files were missing locally
- 9,352 BBBC021 rows skipped because MoA labels were missing for the compound/concentration pair

### Baseline 4 Molecule + Protein Table

Output:

```text
data/processed/baseline_4_molecule_protein.csv
```

Status:

- 46,915 rows
- 42,033 unique molecules
- 12 targets
- no rows skipped for missing protein

This is the strongest aligned multimodal table currently available.

### Baseline 5 Molecule + Image Table

Output:

```text
data/processed/baseline_5_molecule_image.csv
```

Status:

- 300 rows
- 6 unique compounds
- 3 MoA classes

This is valid for a small molecule-image proof of concept, but it is much smaller than the ChEMBL molecule-protein dataset.

### Baseline 6 and 7 Blockers

Baseline 6, protein + image:

```text
blocked: BBBC021 image rows do not currently contain target/protein annotations.
```

Required alignment:

```text
image row -> compound -> known target -> UniProt/protein sequence
```

Baseline 7, molecule + protein + image:

```text
blocked: exact SMILES overlap between molecule-protein and molecule-image tables is 0.
```

Current alignment counts:

| Item | Count |
|---|---:|
| Molecule-protein SMILES | 42,033 |
| Molecule-image SMILES | 6 |
| Exact SMILES overlap | 0 |

## Model Code Status

Reusable model components are implemented under:

```text
src/multimodal_datapipeline/models/
```

| File | Status |
|---|---|
| `molecule_encoder.py` | RDKit Morgan fingerprint featurizer + MLP molecule encoder |
| `protein_encoder.py` | ESM-2 sequence encoder + projection layer |
| `image_encoder.py` | DINOv2 image encoder + projection layer |
| `fusion.py` | concatenation fusion head for two or three modalities |

Important implementation detail:

- The model components exist and are reusable.
- The runnable baseline scripts do not all use these components yet.
- Baseline 1 uses a lightweight hashed SMILES n-gram logistic regression.
- Baseline 3 uses a small custom microscopy CNN, not DINOv2.
- Baseline 4, 5, 6, and 7 training scripts are guarded placeholders.

## Baseline Script Status

Baseline scripts are under:

```text
scripts/baselines/
```

| Baseline | Data Curation | Validation | Training | Current Status |
|---|---|---|---|---|
| 1 Molecule-only | Done | Done | Done | Runnable and run |
| 2 Protein-only | Done | Done | Done | Runnable and run |
| 3 Image-only | Done | Done | Done | Runnable and run |
| 4 Molecule + protein | Done | Done | Placeholder | Data ready, training pending |
| 5 Molecule + image | Done | Done | Placeholder | Data ready, training pending |
| 6 Protein + image | Blocked | Guarded | Placeholder | Blocked by missing alignment |
| 7 Molecule + protein + image | Blocked | Guarded | Placeholder | Blocked by zero SMILES overlap |

## Saved Experiment Results

Saved outputs are under:

```text
multimodal_datapipeline/experiments/
```

### Baseline 1: Molecule-Only

Run folder:

```text
experiments/molecule_only_baseline/
```

Task:

```text
SMILES -> hashed character n-gram features -> logistic regression SGD -> active/inactive
```

Dataset:

- 42,033 unique SMILES
- 34,024 active molecules
- 8,009 inactive molecules

Test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.8134 |
| Precision | 0.8132 |
| Recall | 0.9990 |
| F1 | 0.8965 |
| ROC-AUC | 0.7405 |
| Test examples | 8,407 |

Interpretation:

This baseline is heavily biased toward predicting active molecules. It gives a useful sanity baseline but has weak inactive-class separation.

### Baseline 2: Protein-Only

Run folder:

```text
experiments/baseline_2_protein_only/
```

Task:

```text
protein sequence -> frozen ESM-2 -> MLP -> active_fraction
```

Dataset:

- 12 protein targets
- split: 8 train, 2 validation, 2 test
- device: MPS

Test metrics:

| Metric | Value |
|---|---:|
| MSE | 0.0043 |
| RMSE | 0.0659 |
| MAE | 0.0596 |
| R2 | -44.4220 |

Interpretation:

This is a pipeline sanity check, not a strong scientific result. With only 12 target-level rows, R2 is unstable and should not be used as a thesis claim.

### Baseline 3: Image-Only

Run folder:

```text
experiments/baseline_3_image_only/
```

Task:

```text
3-channel BBBC021 microscopy image -> small CNN -> MoA class
```

Dataset:

- 516 image rows
- 5 MoA classes
- split: 361 train, 52 validation, 103 test
- device: MPS

MoA class counts:

| Class | Rows |
|---|---:|
| Actin disruptors | 48 |
| Aurora kinase inhibitors | 72 |
| DMSO | 144 |
| Microtubule destabilizers | 36 |
| Microtubule stabilizers | 216 |

Test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.6893 |
| Macro precision | 0.8476 |
| Macro recall | 0.7188 |
| Macro F1 | 0.6434 |
| Loss | 0.7443 |

Interpretation:

Baseline 3 is now implemented and has a completed run. The dataset is small and class-imbalanced, but the model learns meaningful image signal.

### Additional Saved Outputs

Other saved folders:

- `baseline_1_smoke_test/`
- `baseline_3_image_only_smoke_test/`
- `molecule_only_baseline_wrapper_test/`
- `molecule_encoder_validation/`

Molecule encoder validation result:

- 52,811 unique SMILES checked
- all 52,811 passed RDKit validation
- 15,532 had stereochemistry
- molecule encoder smoke test passed with embedding shape `[16, 256]`

## Current Gaps

1. Baseline 4 training is not implemented inside `multimodal_datapipeline`.

The molecule + protein table is ready, but `scripts/baselines/baseline_4_molecule_protein/train.py` exits with:

```text
Baseline 4 data is prepared. Next step: train MoleculeEncoder + ProteinEncoder + FusionHead.
```

2. Baseline 5 training is not implemented inside `multimodal_datapipeline`.

The molecule + image table is ready, but `scripts/baselines/baseline_5_molecule_image/train.py` exits with:

```text
Baseline 5 data is prepared. Next step: train MoleculeEncoder + ImageEncoder + FusionHead.
```

3. Baseline 6 is blocked by missing protein-image labels.

BBBC021 rows currently provide compound/image/MoA information, not target/protein annotations.

4. Baseline 7 is blocked by molecule identity alignment.

There is currently no exact SMILES overlap between:

```text
baseline_4_molecule_protein.csv
baseline_5_molecule_image.csv
```

5. Baseline 1 is a weak classical baseline.

It uses hashed SMILES character n-grams and logistic regression, not the RDKit Morgan fingerprint MLP model component.

6. Baseline 3 does not use the DINOv2 model component yet.

The reusable DINOv2 image encoder exists, but the completed Baseline 3 run uses a small CNN.

## Recommended Next Work

Priority 1:

Implement Baseline 4 training inside `multimodal_datapipeline`:

```text
Morgan fingerprint / MoleculeEncoder -> molecule embedding
ESM-2 / ProteinEncoder -> protein embedding
ConcatenationFusion -> active/inactive
```

Use `data/processed/baseline_4_molecule_protein.csv`, scaffold split, and report ROC-AUC, PR-AUC, F1, and balanced accuracy.

Priority 2:

Add a stronger molecule-only neural baseline inside `multimodal_datapipeline` using:

```text
RDKit Morgan fingerprint -> MLP -> active/inactive
```

This will be a fairer comparator for Baseline 4 than the current hashed n-gram logistic regression.

Priority 3:

Implement Baseline 5 molecule-image fusion:

```text
MoleculeEncoder + image CNN/DINOv2 encoder -> fusion -> MoA
```

Keep it as proof of concept because the table has only 300 rows and 6 compounds.

Priority 4:

Do not spend major thesis effort on Baseline 6 or Baseline 7 until biological alignment is fixed.

Required future curation:

```text
BBBC021 compound -> canonical SMILES -> ChEMBL molecule ID/activity -> target -> UniProt
```

If exact overlap remains zero, use a different imaging dataset, a different compound-target source, or similarity-based exploratory matching with clear caveats.
