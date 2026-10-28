# Baseline Experiment Scripts

Each folder contains one baseline experiment family. Keep baseline-specific
curation, validation, training, and wrapper modules inside that baseline folder.

Current layout:

- `baseline_1_molecule_only/`: SMILES -> molecule encoder/features -> prediction
- `baseline_2_protein_only/`: protein sequence/structure -> protein encoder -> prediction
- `baseline_3_image_only/`: BBBC021 image -> image encoder -> prediction
- `baseline_4_molecule_protein/`: molecule + protein fusion
- `baseline_5_molecule_image/`: molecule + image fusion
- `baseline_6_protein_image/`: protein + image fusion
- `baseline_7_molecule_protein_image/`: full proposed multimodal model

All seven baselines are runnable:

- 1-4 on ChEMBL kinases (and Baseline 4 on the Davis benchmark),
- 3 and 5 on BBBC021 MoA (5 adds the molecule to the image phenotype),
- 6 and 7 on the aligned BBBC021-compound / ChEMBL-protein / image table.

Model explanations live in `workflows/interpretability/`.

Realistic splits are shared through `package/multimodal_datapipeline/utils/splits.py`:
scaffold (Baselines 1 and 4), cold target (Baseline 4), held-out compound and
held-out week (Baseline 3).
