# Deep Learning Project Experiment Report

## 1. Project Objective

This project evaluates whether deep learning can improve drug activity prediction using the curated data already available in the research workspace.

The central research question is:

```text
Does adding protein/target information improve drug activity prediction beyond molecule-only prediction?
```

The project currently focuses on ChEMBL-derived molecule and target activity data. The experiments are implemented under:

```text
Mtech_project/deep_learning_project/experiments/
```

The experiments use processed data from:

```text
Mtech_project/multimodal_datapipeline/data/processed/
```

Only the data is reused. The model code, feature extraction, metrics, and experiment scripts are owned by `deep_learning_project`.

## 2. Data Used

### Molecule-Only Data

Input table:

```text
multimodal_datapipeline/data/processed/chembl_molecule_curated.csv
```

Used in:

```text
experiments/experiment_1_molecule_only_mlp.py
```

Summary:

| Item | Count |
|---|---:|
| Unique SMILES | 42,033 |
| Active molecules | 34,024 |
| Inactive molecules | 8,009 |
| Activity threshold | pChEMBL >= 6.0 |

### Protein-Only Data

Input table:

```text
multimodal_datapipeline/data/processed/baseline_2_protein_only.csv
```

Used in:

```text
experiments/experiment_2_protein_only_esm2.py
```

Summary:

| Item | Count |
|---|---:|
| Protein targets | 12 |
| Train rows | 8 |
| Validation rows | 2 |
| Test rows | 2 |

This data is very small, so protein-only results are treated as a sanity check rather than a strong result.

### Molecule-Protein Data

Input table:

```text
multimodal_datapipeline/data/processed/baseline_4_molecule_protein.csv
```

Used in:

```text
experiments/experiment_3_molecule_protein_fusion.py
```

Summary:

| Item | Count |
|---|---:|
| Molecule-target rows | 46,915 |
| Unique SMILES | 42,033 |
| Protein targets | 12 |
| Active rows | 37,261 |
| Inactive rows | 9,654 |

## 3. Feature Engineering

Feature utilities are located in:

```text
experiments/deep_learning_utils/featurizers.py
```

### Molecule Features

Molecules are represented using RDKit Morgan fingerprints:

```text
SMILES -> Morgan fingerprint
```

Configuration:

| Parameter | Value |
|---|---:|
| Radius | 2 |
| Bits | 2048 |
| Chirality | Enabled |

### Protein Features

Two protein representations are used:

1. ESM-2 sequence embeddings in Experiment 2.
2. Hashed amino-acid k-mer vectors in Experiment 3.

Experiment 3 uses:

| Protein feature | Value |
|---|---:|
| Representation | Hashed amino-acid k-mer vector |
| k-mer size | 3 |
| Feature dimension | 1024 |

The k-mer representation is lightweight and fast. It gives a target-context signal without needing to run ESM-2 for every molecule-target row.

## 4. Experiment 1: Molecule-Only MLP

Script:

```text
experiments/experiment_1_molecule_only_mlp.py
```

### Objective

Determine whether molecular structure alone can predict activity.

### Model

```text
SMILES -> Morgan fingerprint -> MLP classifier -> active/inactive
```

Architecture:

```text
Linear -> BatchNorm -> ReLU -> Dropout
Linear -> ReLU -> Dropout
Linear -> activity logit
```

Training:

| Setting | Value |
|---|---:|
| Epochs | 20 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Hidden dimension | 512 |
| Embedding dimension | 256 |
| Dropout | 0.25 |
| Device | MPS |

### Split

| Split | Rows |
|---|---:|
| Train | 29,423 |
| Validation | 4,203 |
| Test | 8,407 |

### Results

| Split | Accuracy | Balanced Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Train | 0.9842 | 0.9897 | 0.9901 | 0.9994 | 0.9999 |
| Validation | 0.8660 | 0.8118 | 0.9158 | 0.8981 | 0.9685 |
| Test | 0.8799 | 0.8210 | 0.9251 | 0.9064 | 0.9692 |

### Interpretation

The molecule-only MLP performs strongly. A test ROC-AUC of `0.9064` and PR-AUC of `0.9692` show that molecular structure alone contains substantial activity signal.

However, the train metrics are much higher than validation and test metrics. This indicates that the model fits the training molecules very strongly and may be partly benefiting from similarity between train and test molecules under the random stratified split.

This experiment is useful as a baseline, but it does not answer target-specific drug-target interaction because repeated SMILES are collapsed into one molecule-level label.

## 5. Experiment 2: Protein-Only ESM-2 Sanity Baseline

Script:

```text
experiments/experiment_2_protein_only_esm2.py
```

### Objective

Check whether the protein encoder pipeline works on the 12 available target-level rows.

### Model

```text
Protein sequence -> frozen ESM-2 -> MLP regressor -> active_fraction
```

Backbone:

```text
facebook/esm2_t6_8M_UR50D
```

Training:

| Setting | Value |
|---|---:|
| Epochs | 25 |
| Batch size | 2 |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Embedding dimension | 256 |
| Hidden dimension | 128 |
| Dropout | 0.2 |
| Device | MPS |

### Results

| Split | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Train | 0.0769 | 0.0673 | -0.9263 |
| Validation | 0.0943 | 0.0814 | -3.7979 |
| Test | 0.0459 | 0.0457 | -21.1030 |

### Interpretation

This experiment confirms that the ESM-2-based protein model can run end to end. It should not be used as a strong scientific result.

The reason is data size:

```text
Only 12 protein-level rows are available.
```

With only 8 training proteins and 2 test proteins, R2 is unstable. The negative R2 does not necessarily mean ESM-2 is poor; it mainly means the current protein-only supervised setup is too small for reliable regression.

This experiment should be described as a sanity baseline.

## 6. Experiment 3: Molecule + Protein Fusion

Script:

```text
experiments/experiment_3_molecule_protein_fusion.py
```

### Objective

Test the main research question:

```text
Does adding protein/target sequence context improve molecule-target activity prediction?
```

### Compared Models

Two models are trained on the same molecule-target rows.

#### Model A: Row-Level Molecule-Only Model

```text
Morgan fingerprint -> MLP -> active/inactive
```

This model receives the molecule but not the target sequence.

#### Model B: Molecule + Protein Fusion Model

```text
Morgan fingerprint -> molecule MLP ----------------\
                                                     -> fusion MLP -> active/inactive
Protein k-mer vector -> protein MLP ----------------/
```

This model receives both the molecule and protein/target sequence context.

### Split

Experiment 3 uses scaffold split by default.

```text
Split type: scaffold
```

Split summary:

| Split | Rows |
|---|---:|
| Train | 32,840 |
| Validation | 4,692 |
| Test | 9,383 |

Test label balance:

| Label | Count |
|---|---:|
| Inactive | 1,490 |
| Active | 7,893 |

All 12 targets appear in train, validation, and test. The split tests generalization to unseen molecular scaffolds rather than unseen targets.

### Overall Test Results

| Model | Accuracy | Balanced Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Molecule only | 0.8270 | 0.7853 | 0.8917 | 0.8720 | 0.9703 |
| Molecule + protein fusion | 0.8565 | 0.7969 | 0.9121 | 0.8837 | 0.9729 |

### Improvement From Protein Context

| Metric | Improvement |
|---|---:|
| ROC-AUC | +0.0117 |
| PR-AUC | +0.0027 |
| F1 | +0.0204 |
| Balanced accuracy | +0.0116 |
| Accuracy | +0.0295 |

### Main Insight

Adding protein/target sequence context improves activity prediction on scaffold split.

The improvement is modest but meaningful:

```text
Molecule-only ROC-AUC:          0.8720
Molecule + protein ROC-AUC:    0.8837
Delta:                        +0.0117
```

This supports the project hypothesis that target context adds useful biological signal beyond molecule structure alone.

## 7. Per-Target Behavior in Experiment 3

The fusion model does not improve every target equally. This is important because the dataset is target-imbalanced and some targets have very few inactive examples.

### Molecule-Only Per-Target ROC-AUC

| Target | ROC-AUC | F1 | Balanced Accuracy | Test Rows |
|---|---:|---:|---:|---:|
| CHEMBL2842 | 0.9485 | 0.9608 | 0.9142 | 628 |
| CHEMBL1862 | 0.9292 | 0.8534 | 0.8390 | 377 |
| CHEMBL5145 | 0.8939 | 0.9393 | 0.8056 | 684 |
| CHEMBL203 | 0.8824 | 0.8708 | 0.8079 | 2,293 |
| CHEMBL4040 | 0.8554 | 0.9638 | 0.6132 | 1,351 |
| CHEMBL4816 | 0.8495 | 0.9023 | 0.6978 | 79 |
| CHEMBL4282 | 0.8398 | 0.9310 | 0.7201 | 959 |
| CHEMBL267 | 0.8199 | 0.6570 | 0.6817 | 612 |
| CHEMBL279 | 0.7924 | 0.8489 | 0.7103 | 1,778 |
| CHEMBL3385 | 0.7683 | 0.8244 | 0.6506 | 81 |
| CHEMBL301 | 0.7399 | 0.7645 | 0.6589 | 222 |
| CHEMBL2431 | 0.7395 | 0.9418 | 0.6466 | 319 |

### Molecule + Protein Fusion Per-Target ROC-AUC

| Target | ROC-AUC | F1 | Balanced Accuracy | Test Rows |
|---|---:|---:|---:|---:|
| CHEMBL2842 | 0.9603 | 0.9637 | 0.9100 | 628 |
| CHEMBL1862 | 0.9257 | 0.9100 | 0.8529 | 377 |
| CHEMBL5145 | 0.9052 | 0.9531 | 0.8321 | 684 |
| CHEMBL203 | 0.8994 | 0.8896 | 0.8291 | 2,293 |
| CHEMBL4816 | 0.8532 | 0.8571 | 0.8389 | 79 |
| CHEMBL267 | 0.8272 | 0.8231 | 0.7595 | 612 |
| CHEMBL3385 | 0.8205 | 0.9231 | 0.7317 | 81 |
| CHEMBL4282 | 0.8203 | 0.9073 | 0.7164 | 959 |
| CHEMBL4040 | 0.8158 | 0.9745 | 0.5764 | 1,351 |
| CHEMBL279 | 0.8048 | 0.8786 | 0.7222 | 1,778 |
| CHEMBL301 | 0.8028 | 0.8719 | 0.6667 | 222 |
| CHEMBL2431 | 0.7506 | 0.9097 | 0.6328 | 319 |

### Per-Target Interpretation

Protein context improves several targets, including:

- CHEMBL203
- CHEMBL267
- CHEMBL279
- CHEMBL301
- CHEMBL3385
- CHEMBL5145

Some targets do not improve or slightly decrease, especially targets with heavy class imbalance or very few inactive examples. For example, CHEMBL4040 has 1,330 active rows and only 21 inactive rows in the test split, so balanced accuracy and ROC-AUC are sensitive to small prediction changes.

## 8. Experiment 4: Molecule 3D Point-Cloud Classifier

Script:

```text
experiments/experiment_4_molecule_3d_pointcloud.py
```

### Objective

Add a direct 3D Vision & Geometry course-topic experiment to the project:

```text
Can approximate 3D molecular geometry predict activity?
```

### Course Syllabus Alignment

This experiment uses topics from Unit III:

- 3D Point Cloud
- Volumetric / structure representation
- Euclidean geometry through centered and scale-normalized xyz coordinates
- Rotation handling through optional random rotation augmentation

### Model

```text
SMILES -> RDKit ETKDG 3D conformer -> atom point cloud -> PointNet-style classifier -> active/inactive
```

Each molecule is converted into a fixed-size atom point cloud. Each point contains:

- normalized xyz coordinates
- atomic number
- degree
- formal charge
- hydrogen count
- aromatic/ring flags
- hybridization flags

Run command:

```bash
multimodal_datapipeline/.venv/bin/python deep_learning_project/experiments/experiment_4_molecule_3d_pointcloud.py --augment-rotation
```

The default row cap is:

```text
--max-rows 3000
```

This is intentional because 3D conformer generation is much slower than Morgan fingerprint generation.

### Interpretation

Experiment 4 is the cleanest answer to the 3D Vision & Geometry syllabus requirement. Experiment 3 is a molecule-protein deep learning experiment, but Experiment 4 explicitly uses 3D point clouds and geometric normalization.

After running it fully, compare its test ROC-AUC and PR-AUC against Experiment 1 to evaluate whether generated 3D geometry adds useful signal beyond 2D molecular fingerprints.

## 9. Overall Findings

### Finding 1: Molecule structure is a strong signal

Experiment 1 shows that Morgan fingerprints plus an MLP can predict activity well:

```text
Test ROC-AUC: 0.9064
Test PR-AUC: 0.9692
```

This establishes a strong molecule-only baseline.

### Finding 2: Protein-only supervised learning is underpowered

Experiment 2 is not reliable as a standalone scientific result because it has only 12 target-level rows.

The experiment is still useful because it confirms the ESM-2 pipeline works.

### Finding 3: Protein context improves target-aware prediction

Experiment 3 is the most important result:

```text
Molecule-only scaffold ROC-AUC:       0.8720
Molecule + protein scaffold ROC-AUC: 0.8837
```

The model gains performance by including protein sequence context. This supports the broader project direction: multimodal or target-aware models are more informative than molecule-only models.

### Finding 4: Experiment 4 aligns the project with 3D Vision & Geometry

Experiment 4 uses an atom point cloud generated from 3D molecular conformers. This creates a direct link to Unit III of the 3D Vision & Geometry syllabus.

## 10. Limitations

### 1. Experiment 1 uses molecule-level labels

Experiment 1 collapses repeated SMILES across targets into one molecule-level label. This is acceptable for a molecule-only baseline, but it loses target-specific information.

For drug-target interaction, the better formulation is:

```text
(molecule, target) -> active/inactive
```

Experiment 3 uses that better formulation.

### 2. Experiment 2 has too few samples

Only 12 protein targets are available. This is too small for reliable supervised protein-only training.

### 3. Experiment 3 uses protein k-mer features, not AlphaFold structure yet

The current fusion model uses protein sequence k-mer features. It does not yet use AlphaFold contact maps or residue graphs.

This means Experiment 3 supports target-aware prediction, but not yet structure-aware prediction.

### 4. Scaffold split still contains all targets in train and test

The scaffold split tests unseen chemical scaffolds, but not unseen protein targets.

A stronger generalization test would be:

```text
cold-target split
```

### 5. Class imbalance remains substantial

The ChEMBL activity labels are imbalanced toward active rows. This is why PR-AUC is high and balanced accuracy is more informative than accuracy alone.

### 6. Experiment 4 uses generated conformers

The 3D coordinates are approximate RDKit conformers, not experimentally solved ligand poses inside a protein pocket. This is still valid for a 3D point-cloud experiment, but it should not be presented as ligand-protein docking.

## 11. Recommended Next Experiments

### Priority 1: Cold-Target Evaluation

Run:

```bash
multimodal_datapipeline/.venv/bin/python deep_learning_project/experiments/experiment_3_molecule_protein_fusion.py --split cold_target --out-dir deep_learning_project/experiments/experiment_3_cold_target
```

Purpose:

```text
Test whether molecule-protein fusion generalizes to unseen targets.
```

This is a stronger biological generalization experiment than scaffold split alone.

### Priority 2: Full Experiment 4 Run

Run:

```bash
multimodal_datapipeline/.venv/bin/python deep_learning_project/experiments/experiment_4_molecule_3d_pointcloud.py --augment-rotation
```

Purpose:

```text
Produce full metrics for the 3D point-cloud course-topic experiment.
```

### Priority 3: AlphaFold Contact-Map Model

Add an experiment:

```text
SMILES Morgan fingerprint + AlphaFold contact map -> fusion model
```

This would directly test the structure-based part of the research proposal.

Recommended model:

```text
Morgan fingerprint -> molecule MLP
AlphaFold contact map -> CNN
concat -> fusion MLP -> active/inactive
```

### Priority 4: Compare Three Protein Representations

Compare:

1. Protein k-mer vector
2. ESM-2 embedding
3. AlphaFold contact map

This would answer:

```text
Which protein representation contributes the most useful target context?
```

### Priority 4: Save a Summary Table for Thesis

Create a final table:

| Experiment | Input | Split | ROC-AUC | PR-AUC | F1 | Balanced Accuracy |
|---|---|---|---:|---:|---:|---:|
| Exp 1 | Molecule only | Random | 0.9064 | 0.9692 | 0.9251 | 0.8210 |
| Exp 3A | Molecule only | Scaffold | 0.8720 | 0.9703 | 0.8917 | 0.7853 |
| Exp 3B | Molecule + protein k-mer | Scaffold | 0.8837 | 0.9729 | 0.9121 | 0.7969 |

## 11. Conclusion

The experiments show a clear progression:

```text
Molecule-only learning works well.
Protein-only learning is underpowered with only 12 targets.
Molecule + protein fusion gives better scaffold-split performance than molecule-only prediction.
```

The most important result is Experiment 3:

```text
Adding protein sequence context improves molecule-target activity prediction.
```

This is an impactful first insight because it directly supports the claim that target-aware deep learning can improve drug activity prediction over molecule-only models.

The next major step should be to replace or augment the protein k-mer vector with AlphaFold-derived structural features. That would move the project from target-aware prediction toward true structure-aware drug discovery.
