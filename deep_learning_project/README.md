# Deep Learning Project

## Use Case

**Structure-aware drug activity prediction using molecular SMILES and AlphaFold protein structures.**

This project uses the data already curated under:

```text
../multimodal_datapipeline/data/processed/
../multimodal_datapipeline/dataset_pipeline_output/
```

The primary deep-learning objective is to predict whether a molecule is active against a target protein by combining:

- molecule information from ChEMBL SMILES
- protein sequence and AlphaFold structure information
- supervised activity labels from curated ChEMBL IC50 / pChEMBL data

## Why This Use Case Fits The Available Data

The strongest aligned dataset available right now is the molecule-protein activity table:

```text
../multimodal_datapipeline/data/processed/baseline_4_molecule_protein.csv
```

It contains:

- 46,915 molecule-target rows
- 42,033 unique molecules
- 12 protein targets
- 12 corresponding AlphaFold PDB structures
- binary activity labels
- median pChEMBL values

This makes it suitable for a deep-learning drug-target interaction experiment. The BBBC021 image data can be used as a secondary image-only or molecule-image extension, but the full molecule-protein-image use case is not yet aligned because the current processed data has no exact SMILES overlap between ChEMBL and BBBC021.

## Project Structure

```text
deep_learning_project/
  README.md
  configs/
    project_paths.yaml
  data/
    README.md
  docs/
    data_catalog.md
    experiment_plan.md
    usecase_design.md
  experiments/
    experiment_1_molecule_only_mlp.py
    experiment_2_protein_only_esm2.py
    experiment_3_molecule_protein_fusion.py
    experiment_4_molecule_3d_pointcloud.py
    deep_learning_utils/
    experiment_1_molecule_only_mlp/
    experiment_2_protein_only_esm2/
    experiment_3_molecule_protein_fusion/
    .gitkeep
  reports/
    .gitkeep
```

## Recommended First Model

Start with a two-encoder neural model:

```text
SMILES / molecule graph -> molecule encoder ----\
                                                 -> fusion MLP -> activity prediction
Protein sequence / AlphaFold features -> encoder /
```

Recommended first implementation:

- Molecule encoder: Morgan fingerprint + MLP
- Protein encoder: ESM-2 embedding or AlphaFold-derived contact-map features + MLP/CNN
- Fusion: concatenation + MLP
- Task: binary active/inactive classification
- Metrics: ROC-AUC, PR-AUC, F1, balanced accuracy

## Phase 2 Scripts

Experiment 1:

```bash
python experiments/experiment_1_molecule_only_mlp.py
```

This trains:

```text
SMILES -> Morgan fingerprint -> MLP -> active / inactive
```

Outputs:

```text
experiments/experiment_1_molecule_only_mlp/metrics.json
experiments/experiment_1_molecule_only_mlp/test_predictions.csv
```

Experiment 2:

```bash
python experiments/experiment_2_protein_only_esm2.py
```

This trains:

```text
protein sequence -> frozen ESM-2 -> MLP -> active_fraction
```

The first run may need internet access to download `facebook/esm2_t6_8M_UR50D`, unless the model is already cached or `--model-name` points to a local model directory.

Outputs:

```text
experiments/experiment_2_protein_only_esm2/metrics.json
experiments/experiment_2_protein_only_esm2/test_predictions.csv
```

Experiment 3:

```bash
python experiments/experiment_3_molecule_protein_fusion.py
```

This trains two models on the same molecule-target rows:

```text
Morgan fingerprint -> MLP -> active/inactive
Morgan fingerprint + protein k-mer features -> fusion MLP -> active/inactive
```

Default split:

```text
scaffold
```

The key output is a direct comparison answering whether protein/target context improves activity prediction.

Outputs:

```text
experiments/experiment_3_molecule_protein_fusion/metrics.json
experiments/experiment_3_molecule_protein_fusion/test_predictions.csv
experiments/experiment_3_molecule_protein_fusion/comparison_report.md
```

Experiment 4:

```bash
python experiments/experiment_4_molecule_3d_pointcloud.py --augment-rotation
```

This is the course-aligned 3D Vision & Geometry experiment:

```text
SMILES -> RDKit 3D conformer -> atom point cloud -> PointNet-style classifier -> active/inactive
```

Syllabus topics used:

- Unit III: 3D Point Cloud
- Unit III: Volumetric / structure representation
- Euclidean geometry through centered/scaled xyz coordinates and random rotation augmentation

The default `--max-rows 3000` keeps 3D conformer generation practical. Increase it for a fuller run after the smoke test passes.

Outputs:

```text
experiments/experiment_4_molecule_3d_pointcloud/metrics.json
experiments/experiment_4_molecule_3d_pointcloud/test_predictions.csv
experiments/experiment_4_molecule_3d_pointcloud/comparison_report.md
experiments/experiment_4_molecule_3d_pointcloud/conformer_failures.csv
```

## Secondary Extensions

After the primary model works:

1. Add AlphaFold contact-map CNN features.
2. Add AlphaFold residue-graph GNN features.
3. Run scaffold split and leave-one-target-out evaluation.
4. Build an image-only BBBC021 MoA classifier.
5. Build a small molecule-image MoA classifier using the existing 300-row table.
