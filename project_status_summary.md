# M.Tech Project Status Summary

Last updated: 2026-09-15

## Overall

`multimodal_datapipeline/` is organized as a staged multimodal drug-discovery data pipeline and baseline-experiment project.

The current project has two complementary experiment areas:

- `Mtech_project/multimodal_datapipeline/`: data acquisition, processed tables, reusable model components, workflow baselines, and saved pipeline results.
- `deep_learning_project/`: independent deep-learning experiments that reuse the processed pipeline tables for stronger neural baselines and comparative reports.
- `3D_CV_Geometry/`: 3D molecular geometry experiment artifacts, including a PointNet-style 3D molecule model.

The project currently supports:

- dataset ingestion from ChEMBL, AlphaFold, BBBC021, and optional HTML scraping
- cytoskeleton-specific ChEMBL acquisition
- processed molecule, protein, image, molecule-protein, and molecule-image tables
- reusable molecule, protein, image, and fusion model components
- completed molecule-only, protein-only, image-only, molecule-protein fusion, and molecule 3D experiments
- image-only model comparison runs, including CNN variants and DINOv2 linear probing
- Pants-based Python target ownership and dependency lockfile generation

The strongest aligned multimodal table currently available is the molecule + protein table. Full molecule + protein + image modeling is still blocked by compound identity alignment between ChEMBL and BBBC021.

## Current Directory Structure

```text
Mtech_project/
├── multimodal_datapipeline/
│   ├── 3rdparty/python/
│   ├── configs/
│   ├── data/
│   ├── dataset_pipeline_output/
│   ├── package/multimodal_datapipeline/
│   ├── reports/
│   ├── results/
│   ├── workflows/baselines/
│   ├── pants.toml
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── README.md
├── experiment_plan.txt
├── experiments_conducted.md
└── project_status_summary.md

deep_learning_project/
├── configs/
├── docs/
├── experiments/
├── reports/
├── requirements.txt
└── README.md

3D_CV_Geometry/
└── 3D_computer_vision_and_geometry/
```

## Environment And Tooling

Project virtual environment:

```text
Mtech_project/multimodal_datapipeline/multimodal_venv/
```

Install the pipeline project locally:

```bash
cd Mtech_project/multimodal_datapipeline
source multimodal_venv/bin/activate
python -m pip install -e .
```

Important CLI entrypoints:

```bash
multimodal-datapipeline --help
mmdp-dataset --help
mmdp-baseline-1
mmdp-baseline-2
mmdp-baseline-3
mmdp-baseline-4
mmdp-baseline-5
mmdp-baseline-6
mmdp-baseline-7
```

Pants config and lockfile:

```text
Mtech_project/multimodal_datapipeline/pants.toml
Mtech_project/multimodal_datapipeline/3rdparty/python/default.lock
```

Generate or refresh the lockfile:

```bash
cd Mtech_project/multimodal_datapipeline
pants generate-lockfiles
```

## Data Acquisition Status

Unified acquisition entrypoint:

```bash
cd Mtech_project/multimodal_datapipeline
python data/acquire_data.py --help
```

Implemented acquisition modules:

| Module | Purpose |
|---|---|
| `data/acquire_data.py` | unified data acquisition CLI and cytoskeleton ChEMBL subcommand |
| `data/pipelines/dataset_pipeline.py` | master ChEMBL, AlphaFold, BBBC021, scraping orchestration |
| `data/sources/chembl.py` | ChEMBL activity and target lookup helpers |
| `data/sources/alphafold.py` | AlphaFold metadata and PDB download helpers |
| `data/sources/bbbc021.py` | BBBC021 metadata/image ZIP download and extraction helpers |
| `data/sources/scrape.py` | optional HTML table scraping helper |

Main acquisition command:

```bash
python data/acquire_data.py --download-missing-phase1
```

Other supported acquisition commands:

```bash
python data/acquire_data.py --chembl-target CHEMBL203 --chembl-standard-type IC50 --chembl-max-records 1000
python data/acquire_data.py --alphafold-ids P00533 P31749 P15056
python data/acquire_data.py --download-missing-phase1 --download-bbbc021-images --bbbc021-extract
python data/acquire_data.py cytoskeleton-chembl
```

Current ingestion outputs:

| Dataset | Status | Output |
|---|---|---|
| ChEMBL multi-target activities | Done | `dataset_pipeline_output/chembl/activities_multitarget.csv` |
| ChEMBL per-target activities | Done | `dataset_pipeline_output/chembl/activities_by_target/` |
| ChEMBL target mapping | Done | `dataset_pipeline_output/chembl/target_mapping.csv` |
| ChEMBL single-target activities | Present | `dataset_pipeline_output/chembl/activities.csv` |
| Cytoskeleton ChEMBL activities | Present | `dataset_pipeline_output/chembl/cytoskeleton_activities.csv` |
| AlphaFold structures | Done | `dataset_pipeline_output/alphafold/structures/` |
| AlphaFold metadata | Done | `dataset_pipeline_output/alphafold/metadata.csv` |
| BBBC021 metadata | Done | `dataset_pipeline_output/bbbc021/*.csv` |
| BBBC021 image ZIPs | Present | `dataset_pipeline_output/bbbc021/zips/` |
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
Mtech_project/multimodal_datapipeline/data/processed/
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
- applies drug-like filters for heavy atoms, molecular weight, logP, HBD, and HBA
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

### Baseline Processed Tables

Baseline 2 protein-only:

- output: `data/processed/baseline_2_protein_only.csv`
- 12 target rows
- 12 rows with extracted protein sequence
- all target-level binary labels are active, so the training script uses `active_fraction` regression

Baseline 3 image-only:

- output: `data/processed/baseline_3_image_only.csv`
- 516 usable 3-channel microscopy rows
- 8 unique compounds
- 5 MoA classes
- 3,332 BBBC021 rows skipped because image channel files were missing locally
- 9,352 BBBC021 rows skipped because MoA labels were missing for the compound/concentration pair

Baseline 4 molecule + protein:

- output: `data/processed/baseline_4_molecule_protein.csv`
- 46,915 rows
- 42,033 unique molecules
- 12 targets
- no rows skipped for missing protein
- strongest aligned multimodal table currently available

Baseline 5 molecule + image:

- output: `data/processed/baseline_5_molecule_image.csv`
- 300 rows
- 6 unique compounds
- 3 MoA classes
- valid for a small molecule-image proof of concept

Baseline 6 protein + image:

```text
blocked: BBBC021 image rows do not currently contain target/protein annotations.
```

Required alignment:

```text
image row -> compound -> known target -> UniProt/protein sequence
```

Baseline 7 molecule + protein + image:

```text
blocked: exact SMILES overlap between molecule-protein and molecule-image tables is 0.
```

Current alignment counts:

| Item | Count |
|---|---:|
| Molecule-protein SMILES | 42,033 |
| Molecule-image SMILES | 6 |
| Exact SMILES overlap | 0 |

## Package Code Status

Reusable package code is under:

```text
Mtech_project/multimodal_datapipeline/package/multimodal_datapipeline/
```

Model components:

| File | Status |
|---|---|
| `models/molecule_encoder.py` | RDKit Morgan fingerprint featurizer + MLP molecule encoder |
| `models/protein_encoder.py` | ESM-2 sequence encoder + projection layer |
| `models/image_encoder.py` | DINOv2 image encoder + projection layer |
| `models/fusion.py` | concatenation fusion head for two or three modalities |

Utility components:

| File | Status |
|---|---|
| `utils/io.py` | filesystem, CSV, JSON, and table-writing helpers |
| `utils/paths.py` | project/package/workflow path helpers |
| `utils/baseline_launcher.py` | installed console-entrypoint launcher for baseline training scripts |

Important implementation details:

- Baseline 1 in `multimodal_datapipeline` is a classical hashed SMILES n-gram logistic-regression baseline.
- `deep_learning_project/experiments/experiment_1_molecule_only_mlp.py` adds a stronger RDKit Morgan fingerprint MLP molecule-only model.
- Baseline 3 now has both the original single-run CNN result and a broader image-only suite with CNN variants and DINOv2 linear probing.
- `workflows/baselines/baseline_4_molecule_protein/train.py` now contains an implemented molecule + protein training path using Morgan fingerprints, frozen ESM-2 embeddings, and concatenation fusion.
- Baseline 5 training remains a placeholder.
- Baselines 6 and 7 remain blocked by data alignment.

## Workflow Script Status

Baseline scripts are under:

```text
Mtech_project/multimodal_datapipeline/workflows/baselines/
```

| Baseline | Data Curation | Validation | Training | Current Status |
|---|---|---|---|---|
| 1 Molecule-only | Done | Done | Done | Runnable and run |
| 2 Protein-only | Done | Done | Done | Runnable and run |
| 3 Image-only | Done | Done | Done | Runnable and run; image suite also completed |
| 4 Molecule + protein | Done | Done | Implemented | Data ready; workflow training script implemented; independent fusion experiment completed |
| 5 Molecule + image | Done | Done | Placeholder | Data ready, training pending |
| 6 Protein + image | Blocked | Guarded | Placeholder | Blocked by missing protein-image alignment |
| 7 Molecule + protein + image | Blocked | Guarded | Placeholder | Blocked by zero SMILES overlap |

Direct script pattern:

```bash
python workflows/baselines/baseline_1_molecule_only/curate_data.py
python workflows/baselines/baseline_1_molecule_only/validate_smiles.py
python workflows/baselines/baseline_1_molecule_only/train.py
```

Baseline 4 direct run:

```bash
python workflows/baselines/baseline_4_molecule_protein/train.py
```

## Saved Pipeline Results

Saved pipeline outputs are under:

```text
Mtech_project/multimodal_datapipeline/results/
```

Current saved result folders:

| Folder | Meaning |
|---|---|
| `molecule_encoder_validation/` | RDKit and molecule encoder input validation |
| `molecule_only_baseline/` | Baseline 1 molecule-only classical training output |
| `baseline_2_protein_only/` | Baseline 2 protein-only ESM-2 regression output |
| `baseline_3_image_only/` | Baseline 3 original image-only MoA classification output |
| `baseline_3_image_only_smoke/` | quick image-only smoke run |
| `baseline_3_image_only_suite_smoke/` | quick image-model suite smoke runs |
| `baseline_3_image_only_full/` | full image-model comparison suite |

### Molecule Encoder Validation

Generated by:

```bash
python workflows/baselines/baseline_1_molecule_only/validate_smiles.py
```

Result:

- 52,811 unique SMILES checked
- all 52,811 passed RDKit validation
- 15,532 had stereochemistry
- no unassigned atom or bond stereochemistry
- molecule encoder smoke test passed with embedding shape `[16, 256]`

This is a data-quality validation result, not a model-training result.

### Baseline 1: Molecule-Only Classical

Run folder:

```text
results/molecule_only_baseline/
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

This is a useful sanity baseline, but it is biased toward predicting active molecules and has weak inactive-class separation.

### Baseline 2: Protein-Only

Run folder:

```text
results/baseline_2_protein_only/
```

Task:

```text
protein sequence -> frozen ESM-2 -> MLP -> active_fraction
```

Dataset:

- 12 protein targets
- split: 8 train, 2 validation, 2 test
- model: `facebook/esm2_t6_8M_UR50D`

Test metrics:

| Metric | Value |
|---|---:|
| MSE | 0.0043 |
| RMSE | 0.0659 |
| MAE | 0.0596 |
| R2 | -44.4220 |

Interpretation:

This is a pipeline sanity check, not a strong scientific result. With only 12 target-level rows, R2 is unstable and should not be used as a thesis claim.

### Baseline 3: Image-Only Original Run

Run folder:

```text
results/baseline_3_image_only/
```

Task:

```text
3-channel BBBC021 microscopy image -> small CNN -> MoA class
```

Dataset:

- 516 image rows
- 5 MoA classes
- split: 361 train, 52 validation, 103 test

Test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.6893 |
| Macro precision | 0.8476 |
| Macro recall | 0.7188 |
| Macro F1 | 0.6434 |
| Loss | 0.7443 |

Interpretation:

Baseline 3 is implemented and has a completed run. The dataset is small and class-imbalanced, but the model learns meaningful image signal.

### Baseline 3: Full Image Model Suite

Run folder:

```text
results/baseline_3_image_only_full/
```

Task:

```text
BBBC021 3-channel microscopy image -> image model -> MoA class
```

Completed models:

| Model | Device | Accuracy | Macro F1 | Loss |
|---|---|---:|---:|---:|
| `tiny_cnn` | CPU | 0.7497 | 0.6590 | 0.7455 |
| `small_cnn` | CPU | 0.6680 | 0.5685 | 1.1316 |
| `small_cnn_augmented` | CPU | 0.5370 | 0.4887 | 1.6962 |
| `dinov2_linear` | MPS | 0.8262 | 0.7181 | 0.5795 |

Interpretation:

DINOv2 linear probing is currently the strongest image-only result. The augmented small CNN underperformed in this run, likely because the dataset is small and augmentation may have introduced harder variation than the model could absorb.

## Deep Learning Experiments

Independent experiment scripts and outputs are under:

```text
deep_learning_project/experiments/
```

### Experiment 1: Molecule-Only MLP

Run folder:

```text
deep_learning_project/experiments/experiment_1_molecule_only_mlp/
```

Task:

```text
RDKit Morgan fingerprint -> MLP -> active/inactive
```

Dataset:

- input: `data/processed/chembl_molecule_curated.csv`
- fingerprint: 2048-bit Morgan fingerprint, radius 2
- model: MLP classifier
- test examples: 8,407

Test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.8701 |
| Balanced accuracy | 0.8324 |
| Precision | 0.9432 |
| Recall | 0.8933 |
| Specificity | 0.7715 |
| F1 | 0.9176 |
| ROC-AUC | 0.9128 |
| PR-AUC | 0.9731 |

Interpretation:

This is a much stronger molecule-only neural baseline than the original hashed n-gram logistic-regression baseline.

### Experiment 2: Protein-Only ESM-2

Run folder:

```text
deep_learning_project/experiments/experiment_2_protein_only_esm2/
```

Task:

```text
protein sequence -> frozen ESM-2 -> MLP -> active_fraction
```

Dataset:

- input: `data/processed/baseline_2_protein_only.csv`
- 12 target-level rows
- model: `facebook/esm2_t6_8M_UR50D`

Test metrics:

| Metric | Value |
|---|---:|
| MSE | 0.0014 |
| RMSE | 0.0371 |
| MAE | 0.0368 |
| R2 | -13.3835 |

Interpretation:

This is still a sanity baseline because the protein-only table has only 12 rows.

### Experiment 3: Molecule + Protein Fusion

Run folder:

```text
deep_learning_project/experiments/experiment_3_molecule_protein_fusion/
```

Question:

```text
Does adding protein/target sequence context improve row-level activity prediction?
```

Models compared:

```text
1. molecule_only_pair:
   Morgan fingerprint -> MLP -> active/inactive

2. molecule_protein_kmer_fusion:
   Morgan fingerprint -> MLP --------\
                                      -> fusion MLP -> active/inactive
   protein sequence k-mer features -> MLP /
```

Dataset and split:

- input: `data/processed/baseline_4_molecule_protein.csv`
- split: scaffold
- train rows: 32,840
- validation rows: 4,692
- test rows: 9,383
- test label counts: 7,893 active, 1,490 inactive
- all 12 targets are represented in train, validation, and test

Test metrics:

| Model | Accuracy | Balanced accuracy | F1 | ROC-AUC | PR-AUC | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Molecule-only pair | 0.8522 | 0.7853 | 0.9095 | 0.8765 | 0.9706 | 0.6872 |
| Molecule + protein k-mer fusion | 0.8255 | 0.8089 | 0.8893 | 0.8835 | 0.9725 | 0.7846 |

Interpretation:

Adding protein k-mer sequence context gives a modest improvement in ROC-AUC, PR-AUC, balanced accuracy, and inactive-class specificity, while reducing raw accuracy and F1. This is useful evidence that target context helps especially for inactive-class separation, but the gain is modest and should be interpreted with scaffold-split caveats.

### Experiment 4: Molecule 3D Point Cloud

Run folder:

```text
deep_learning_project/experiments/experiment_4_molecule_3d_pointcloud/
```

Task:

```text
SMILES -> RDKit conformer -> atom-level 3D point cloud -> PointNet-style model -> active/inactive
```

Dataset:

- input: `data/processed/chembl_molecule_curated.csv`
- max rows used: 3,000
- max atoms: 64
- test examples: 600
- saved artifacts include `pointnet_3d.pt`, `test_predictions.csv`, `conformer_failures.csv`, `comparison_report.md`, and `metrics_visualization.png`

Test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.7133 |
| Balanced accuracy | 0.6048 |
| Precision | 0.8293 |
| Recall | 0.7974 |
| Specificity | 0.4122 |
| F1 | 0.8130 |
| ROC-AUC | 0.6782 |
| PR-AUC | 0.8791 |

Interpretation:

The 3D PointNet experiment is completed, but it is weaker than the 2D Morgan fingerprint MLP on the current setup. It is best framed as a 3D-computer-vision extension rather than the main drug-discovery baseline.

## Current Gaps

1. Baseline 5 molecule + image training is not implemented.

The molecule + image table is ready, but the table is small:

```text
300 rows, 6 compounds, 3 MoA classes
```

It should be treated as a proof-of-concept fusion experiment rather than a strong thesis result.

2. Baseline 6 is blocked by missing protein-image labels.

BBBC021 rows currently provide compound/image/MoA information, not target/protein annotations.

3. Baseline 7 is blocked by molecule identity alignment.

There is currently no exact SMILES overlap between:

```text
baseline_4_molecule_protein.csv
baseline_5_molecule_image.csv
```

4. Baseline 2 protein-only remains underpowered.

The protein-only table has only 12 target-level rows, so it is useful for checking the ESM-2 pipeline but not for strong model claims.

5. Molecule + protein fusion has been tested with protein k-mer features, while the workflow Baseline 4 script supports frozen ESM-2 embeddings.

The next clean comparison is to run the implemented workflow Baseline 4 ESM-2 fusion and compare it with the completed k-mer fusion experiment.

6. Image-only DINOv2 is strong, but image fusion remains limited by compound overlap.

DINOv2 linear probing performs best among the completed image-only runs, but this does not yet solve the molecule/protein/image alignment problem.

## Recommended Next Work

Priority 1:

Run and report the implemented workflow Baseline 4 ESM-2 fusion model:

```text
Morgan fingerprint / MoleculeEncoder -> molecule embedding
ESM-2 / ProteinEncoder -> protein embedding
ConcatenationFusion -> active/inactive
```

Compare it directly against the completed `experiment_3_molecule_protein_fusion` k-mer fusion model and the molecule-only pair ablation.

Priority 2:

Use `experiment_1_molecule_only_mlp` as the main molecule-only comparator in the thesis/report, because it is much stronger and fairer than the original classical Baseline 1.

Priority 3:

Report the image-only suite with DINOv2 linear probing as the strongest Baseline 3 result:

```text
best image-only result: DINOv2 linear, accuracy 0.8262, macro F1 0.7181
```

Priority 4:

Implement Baseline 5 molecule-image fusion only as a proof of concept:

```text
MoleculeEncoder + image CNN/DINOv2 encoder -> fusion -> MoA
```

Because the table has only 300 rows and 6 compounds, it should not be overclaimed.

Priority 5:

Do not spend major thesis effort on Baseline 6 or Baseline 7 until biological alignment is fixed.

Required future curation:

```text
BBBC021 compound -> canonical SMILES -> ChEMBL molecule ID/activity -> target -> UniProt
```

If exact overlap remains zero, use a different imaging dataset, a different compound-target source, or similarity-based exploratory matching with clear caveats.

## Suggested Report Framing

Main results to emphasize:

1. The data pipeline successfully builds reusable molecule, protein, image, and paired-modality tables from ChEMBL, AlphaFold, and BBBC021.
2. The strongest molecule-only result is the Morgan fingerprint MLP with ROC-AUC 0.9128 and PR-AUC 0.9731.
3. The molecule + protein fusion experiment gives a modest target-context improvement over molecule-only pair prediction on scaffold split, especially in balanced accuracy and specificity.
4. The strongest image-only result is DINOv2 linear probing on BBBC021 with accuracy 0.8262 and macro F1 0.7181.
5. The 3D PointNet experiment is complete but underperforms Morgan fingerprints, making it an exploratory geometry extension.
6. Full molecule + protein + image modeling remains blocked by real data alignment limitations, not by code structure alone.
