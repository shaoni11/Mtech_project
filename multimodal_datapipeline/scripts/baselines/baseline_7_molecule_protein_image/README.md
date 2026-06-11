# Baseline 7: Molecule + Protein + Image

Flow:

```text
SMILES -> molecule encoder --------\
protein sequence -> protein encoder -> fusion -> prediction
cell image -> image encoder --------/
```

This is the proposed full multimodal model.

Current model components:

```text
src/multimodal_datapipeline/models/molecule_encoder.py
src/multimodal_datapipeline/models/protein_encoder.py
src/multimodal_datapipeline/models/image_encoder.py
src/multimodal_datapipeline/models/fusion.py
```

Recommended next script:

```text
train_full_multimodal_baseline.py
```

Important alignment requirement:

```text
BBBC021 compound -> SMILES -> ChEMBL molecule/activity -> target -> protein sequence/structure
```

Do not treat this as the main runnable model until the molecule-image-protein alignment is strong enough.

Commands:

```bash
python scripts/baselines/baseline_7_molecule_protein_image/curate_data.py
python scripts/baselines/baseline_7_molecule_protein_image/validate_inputs.py
python scripts/baselines/baseline_7_molecule_protein_image/train.py
```
