# Baseline 2: Protein Only

Flow:

```text
Protein sequence / structure -> protein encoder -> prediction
```

Current model component:

```text
src/multimodal_datapipeline/models/protein_encoder.py
```

Planned input table:

```text
target_chembl_id
uniprot_id
protein_sequence
alphafold_pdb_path
label / activity value
```

Training task:

```text
protein_sequence -> ESM-2 -> MLP -> active_fraction
```

The current target-level binary labels are all active, so `train.py` uses
`active_fraction` regression by default. AlphaFold structure paths are retained
in the table for later structure/contact-map experiments.

Commands:

```bash
python scripts/baselines/baseline_2_protein_only/curate_data.py
python scripts/baselines/baseline_2_protein_only/validate_inputs.py
python scripts/baselines/baseline_2_protein_only/train.py
```

Outputs:

```text
experiments/baseline_2_protein_only/metrics.json
experiments/baseline_2_protein_only/test_predictions.csv
```
