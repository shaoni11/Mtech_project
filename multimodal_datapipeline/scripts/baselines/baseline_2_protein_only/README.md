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

Recommended next script:

```text
train_protein_only_baseline.py
```

For now, the protein encoder uses ESM-2 sequence embeddings. AlphaFold structure paths can be added later as structure-derived features or contact-map inputs.

