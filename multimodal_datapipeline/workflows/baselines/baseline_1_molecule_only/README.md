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
python workflows/baselines/baseline_1_molecule_only/curate_data.py
python workflows/baselines/baseline_1_molecule_only/validate_smiles.py
python workflows/baselines/baseline_1_molecule_only/train.py
```

Convenience training command:

```bash
cd Mtech_project/multimodal_datapipeline
source .venv/bin/activate
python workflows/baselines/baseline_1_molecule_only/run.py
```

Models and splits:

```bash
# original hashed-SMILES logistic regression, random split (default)
python workflows/baselines/baseline_1_molecule_only/train.py
# same molecule branch + head as Baseline 4, scaffold split
python workflows/baselines/baseline_1_molecule_only/train.py --model fingerprint_mlp --split scaffold --out-dir results/baseline_1_fingerprint_scaffold
python workflows/baselines/baseline_1_molecule_only/train.py --model gnn --split scaffold --out-dir results/baseline_1_gnn_scaffold
```

`--split scaffold` groups molecules by Bemis-Murcko scaffold so no test scaffold
is seen in training. `metrics.json` records the split diagnostics under `split`.

Default input:

```text
data/processed/chembl_molecule_curated.csv
```

Default output:

```text
results/molecule_only_baseline/
```
