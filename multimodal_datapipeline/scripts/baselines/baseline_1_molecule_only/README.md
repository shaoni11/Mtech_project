# Baseline 1: Molecule Only

Flow:

```text
SMILES -> molecule features/encoder -> prediction
```

Current runnable script:

```bash
cd Mtech_project/multimodal_datapipeline
source .venv/bin/activate
python scripts/baselines/baseline_1_molecule_only/run.py
```

This delegates to:

```text
scripts/molecule_only_baseline.py
```

Default input:

```text
data/processed/chembl_molecule_curated.csv
```

Default output:

```text
experiments/molecule_only_baseline/
```

