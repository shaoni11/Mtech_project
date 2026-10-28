# Baseline 6: Protein + Image

Flow:

```text
protein (ESM-2 or AlphaFold structure GNN) \
                                            -> fusion -> active / inactive
compound phenotype (DINOv2 profile) -------/
```

Uses the aligned table built for Baseline 7
(`data/processed/baseline_7_molecule_protein_image.csv`): BBBC021 compound ->
ChEMBL binding activity -> human protein -> AlphaFold structure. `train.py` runs the
Baseline 7 trainer with `--modalities protein,image`; every Baseline 7 option works.

Commands:

```bash
python workflows/baselines/baseline_6_protein_image/curate_data.py
python workflows/baselines/baseline_6_protein_image/validate_inputs.py
python workflows/baselines/baseline_6_protein_image/train.py
python workflows/baselines/baseline_6_protein_image/train.py --protein-encoder structure_gnn --out-dir results/baseline_6_structure
```
