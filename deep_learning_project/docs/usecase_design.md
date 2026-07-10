# Use Case Design

## Title

Structure-Aware Deep Learning for Drug-Target Activity Prediction

## Problem Statement

Given a small molecule and a human protein target, predict whether the molecule is active against the target. The model should learn from chemical structure, protein sequence, and AlphaFold-derived protein structure information.

This is a practical deep-learning use case for drug discovery because it directly supports virtual screening: ranking molecules before expensive wet-lab testing.

## Available Inputs

### Molecule Modality

Source:

```text
../multimodal_datapipeline/data/processed/baseline_4_molecule_protein.csv
```

Fields:

- `curated_smiles`
- `molecule_chembl_ids`
- `median_pchembl`
- `label`

Possible representations:

- Morgan fingerprint
- learned SMILES transformer embedding
- molecular graph using atom and bond features

### Protein Modality

Source:

```text
../multimodal_datapipeline/data/processed/baseline_4_molecule_protein.csv
../multimodal_datapipeline/dataset_pipeline_output/alphafold/structures/
```

Fields and files:

- `target_chembl_id`
- `uniprot_id`
- `protein_sequence`
- `alphafold_pdb_path`
- PDB structures for 12 proteins

Possible representations:

- ESM-2 sequence embedding
- AlphaFold contact map
- residue-distance matrix
- protein graph where residues are nodes and spatial contacts are edges
- pocket-focused residue features

### Labels

Classification:

- `label = 1`: active
- `label = 0`: inactive

Regression:

- `median_pchembl`

## Primary Deep-Learning Task

Binary activity classification:

```text
(molecule, protein) -> active / inactive
```

The first thesis-ready model should compare:

1. molecule-only neural baseline
2. protein-only sanity baseline
3. molecule + protein sequence fusion
4. molecule + AlphaFold structure fusion

The key research question:

```text
Does AlphaFold-derived protein structure improve activity prediction beyond molecule-only and sequence-only baselines?
```

## Recommended Architecture

### Model 1: Baseline Neural Fusion

```text
Morgan fingerprint -> MLP ----------------\
                                           -> concatenation -> MLP -> activity
ESM-2 protein embedding -> projection MLP /
```

This is the easiest strong baseline.

### Model 2: AlphaFold Contact-Map Fusion

```text
Morgan fingerprint -> MLP ----------------\
                                           -> concatenation -> MLP -> activity
AlphaFold contact map -> CNN -------------/
```

This directly introduces structure-based prediction.

### Model 3: AlphaFold Residue-Graph Fusion

```text
Molecular graph -> GNN -------------------\
                                           -> fusion -> activity
Protein residue graph -> GNN -------------/
```

This is the strongest deep-learning version, but it requires more implementation effort.

## Target Proteins

The current AlphaFold set contains 12 proteins:

- ABL1
- EGFR
- AKT1
- AKT2
- AKT3
- BRAF
- MAPK1
- MAPK3
- CDK2
- KDR / VEGFR2
- SRC
- MTOR

This target set is kinase-heavy, which is useful for a focused drug-discovery case study.

## Expected Outputs

For each molecule-target pair:

- predicted activity probability
- predicted class
- optional predicted pChEMBL
- target-specific ranking of candidate molecules

## Practical Application

The use case can be presented as:

> A deep-learning virtual screening system for kinase inhibitors that combines molecular information with AlphaFold-derived protein structure to prioritize candidate compounds for target-specific activity.

