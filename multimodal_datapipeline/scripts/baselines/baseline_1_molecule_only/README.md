# Baseline 1: Molecule Only

Flow:

```text
SMILES -> molecule features/encoder -> prediction
```

Directory contents:

```text
curate_data.py      # raw ChEMBL -> curated molecule table
validate_smiles.py  # RDKit SMILES/fingerprint/encoder validation
train.py            # molecule-only baseline training
run.py              # convenience wrapper for train.py
```

Recommended order:

```bash
cd Mtech_project/multimodal_datapipeline
source .venv/bin/activate
python scripts/baselines/baseline_1_molecule_only/curate_data.py
python scripts/baselines/baseline_1_molecule_only/validate_smiles.py
python scripts/baselines/baseline_1_molecule_only/train.py
```

Convenience training command:

```bash
cd Mtech_project/multimodal_datapipeline
source .venv/bin/activate
python scripts/baselines/baseline_1_molecule_only/run.py
```

Default input:

```text
data/processed/chembl_molecule_curated.csv
```

Default output:

```text
experiments/molecule_only_baseline/
```
