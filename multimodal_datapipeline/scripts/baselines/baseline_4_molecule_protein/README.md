# Baseline 4: Molecule + Protein

Flow:

```text
SMILES -> molecule encoder --------\
                                    -> fusion -> prediction
protein sequence -> protein encoder /
```

Current model components:

```text
src/multimodal_datapipeline/models/molecule_encoder.py
src/multimodal_datapipeline/models/protein_encoder.py
src/multimodal_datapipeline/models/fusion.py
```

Planned input table:

```text
curated_smiles
target_chembl_id
uniprot_id
protein_sequence
median_pchembl
label
```

Recommended next script:

```text
train_molecule_protein_baseline.py
```

