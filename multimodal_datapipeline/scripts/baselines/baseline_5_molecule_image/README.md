# Baseline 5: Molecule + Image

Flow:

```text
SMILES -> molecule encoder ----\
                                -> fusion -> prediction
cell image -> image encoder ----/
```

Current model components:

```text
src/multimodal_datapipeline/models/molecule_encoder.py
src/multimodal_datapipeline/models/image_encoder.py
src/multimodal_datapipeline/models/fusion.py
```

Planned input table:

```text
compound
smiles
image_path
moa_label / phenotype label
```

Recommended next script:

```text
train_molecule_image_baseline.py
```

This baseline should use BBBC021 compound metadata and image paths. It should not depend on ChEMBL target labels unless a reliable compound alignment exists.

Commands:

```bash
python scripts/baselines/baseline_3_image_only/curate_data.py
python scripts/baselines/baseline_5_molecule_image/curate_data.py
python scripts/baselines/baseline_5_molecule_image/validate_inputs.py
python scripts/baselines/baseline_5_molecule_image/train.py
```
