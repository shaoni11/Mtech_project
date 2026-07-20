# Experiment Plan

## Phase 1: Data Audit

Goal:

Verify that all molecule-protein rows have usable SMILES, protein sequences, labels, and AlphaFold PDB paths.

Checks:

- missing SMILES count
- missing protein sequence count
- missing PDB path count
- active/inactive balance by target
- duplicate molecule-target pairs
- target distribution

Deliverable:

```text
reports/data_audit.md
```

## Phase 2: Single-Modality Baselines

### Experiment 1: Molecule-Only Neural Baseline

Task:

```text
SMILES -> active / inactive
```

Model:

- Morgan fingerprint
- MLP classifier

Metrics:

- ROC-AUC
- PR-AUC
- F1
- balanced accuracy

Purpose:

Establish how much signal comes from molecules alone.

### Experiment 2: Protein-Only Sanity Baseline

Task:

```text
protein -> target-level active fraction
```

Model:

- ESM-2 frozen embedding
- MLP regressor

Important caveat:

This is only a sanity experiment because there are 12 target-level examples. It should not be presented as a strong standalone deep-learning result.

## Phase 3: Molecule + Protein Fusion

### Experiment 3: Sequence-Based Fusion

Task:

```text
(SMILES, protein sequence) -> active / inactive
```

Model:

- Morgan fingerprint MLP
- ESM-2 protein embedding
- concatenation fusion
- MLP classifier

Purpose:

Measure whether target context improves molecule-only prediction.

Implementation status:

```text
experiments/experiment_3_molecule_protein_fusion.py
```

The implemented version deliberately avoids importing experiment/model code from
`multimodal_datapipeline`. It uses the same processed data table but owns its
feature extraction and model definitions inside `deep_learning_project/experiments`.

Implemented comparison:

1. row-level molecule-only MLP
2. row-level molecule + protein k-mer fusion MLP

Default split:

```text
scaffold
```

This gives a stronger drug-discovery insight than a random split because it
tests generalization to unseen molecular scaffolds.

### Experiment 4: AlphaFold Contact-Map Fusion

Task:

```text
(SMILES, AlphaFold contact map) -> active / inactive
```

Model:

- Morgan fingerprint MLP
- contact-map CNN
- concatenation fusion
- MLP classifier

Purpose:

Test structure-aware prediction using AlphaFold.

### Experiment 5: AlphaFold Residue-Graph Fusion

Task:

```text
(molecular graph, protein residue graph) -> active / inactive
```

Model:

- molecular GNN
- protein residue GNN
- fusion MLP or cross-attention

Purpose:

Strongest deep-learning architecture for the project.

## Phase 4: Evaluation Splits

Run each major model under:

1. random split
2. scaffold split
3. cold-drug split
4. leave-one-target-out split

Recommended priority:

```text
random split -> scaffold split -> leave-one-target-out
```

## Phase 5: Structure Ablation

Compare:

1. molecule only
2. molecule + protein sequence
3. molecule + AlphaFold contact map
4. molecule + AlphaFold residue graph
5. molecule + pocket-only AlphaFold graph

Main thesis table:

```text
model | input modalities | split | ROC-AUC | PR-AUC | F1 | balanced accuracy
```

## Phase 6: Optional Computer Vision Extension

### Experiment 6: Image-Only BBBC021 MoA Classification

Task:

```text
cell microscopy image -> mechanism of action
```

Model:

- ResNet, EfficientNet, DINOv2, or ViT

Dataset:

```text
../multimodal_datapipeline/data/processed/baseline_3_image_only.csv
```

Caveat:

The dataset is small: 516 usable rows.

### Experiment 7: Molecule + Image MoA Classification

Task:

```text
(SMILES, cell image) -> mechanism of action
```

Dataset:

```text
../multimodal_datapipeline/data/processed/baseline_5_molecule_image.csv
```

Caveat:

Only 300 rows and 6 compounds. Treat this as proof of concept.

## Recommended Final Scope

For a strong deep-learning project, focus the main work on:

1. molecule-only neural baseline
2. molecule + protein sequence fusion
3. molecule + AlphaFold contact-map fusion
4. scaffold split evaluation
5. leave-one-target-out evaluation
6. interpretability using molecular substructure importance and protein contact/residue importance

The image experiments can be included as an extension, but the current data alignment does not yet support a strong full molecule-protein-image model.
