# Baseline Experiment Scripts

Each folder contains one baseline experiment family. Keep baseline-specific
curation, validation, training, and wrapper scripts inside that baseline folder.

Current layout:

- `baseline_1_molecule_only/`: SMILES -> molecule encoder/features -> prediction
- `baseline_2_protein_only/`: protein sequence/structure -> protein encoder -> prediction
- `baseline_3_image_only/`: BBBC021 image -> image encoder -> prediction
- `baseline_4_molecule_protein/`: molecule + protein fusion
- `baseline_5_molecule_image/`: molecule + image fusion
- `baseline_6_protein_image/`: protein + image fusion
- `baseline_7_molecule_protein_image/`: full proposed multimodal model

Baseline 1 is currently the only runnable baseline. The remaining folders define
the planned structure for the next experiments.
