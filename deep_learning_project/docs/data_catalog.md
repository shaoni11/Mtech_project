# Data Catalog

## Primary Dataset

### Molecule + Protein Activity Table

Path:

```text
../multimodal_datapipeline/data/processed/baseline_4_molecule_protein.csv
```

Rows:

```text
46,915
```

Unique molecules:

```text
42,033
```

Unique targets:

```text
12
```

Important columns:

- `curated_smiles`
- `target_chembl_id`
- `uniprot_id`
- `protein_sequence`
- `alphafold_pdb_path`
- `median_pchembl`
- `label`
- `molecule_chembl_ids`

Recommended use:

- primary supervised training table
- molecule-protein activity classification
- molecule-protein pChEMBL regression

## AlphaFold Structures

Metadata:

```text
../multimodal_datapipeline/dataset_pipeline_output/alphafold/metadata.csv
```

PDB directory:

```text
../multimodal_datapipeline/dataset_pipeline_output/alphafold/structures/
```

Available UniProt IDs:

- `P00519`
- `P00533`
- `P12931`
- `P15056`
- `P24941`
- `P27361`
- `P28482`
- `P31749`
- `P31751`
- `P35968`
- `P42345`
- `Q9Y243`

Recommended derived features:

- contact maps
- residue distance matrices
- residue graphs
- pocket-centered residue subgraphs
- confidence-aware masks if pLDDT is extracted from PDB B-factor fields

## Molecule-Only Dataset

Path:

```text
../multimodal_datapipeline/data/processed/chembl_molecule_curated.csv
```

Rows:

```text
46,915 molecule-target rows after aggregation
```

Unique SMILES:

```text
42,033
```

Class balance:

```text
37,261 active
9,654 inactive
```

Recommended use:

- molecule-only baseline
- scaffold split experiments
- molecular encoder validation

## Image-Only Dataset

Path:

```text
../multimodal_datapipeline/data/processed/baseline_3_image_only.csv
```

Rows:

```text
516
```

Unique compounds:

```text
8
```

Unique MoA classes:

```text
5
```

Recommended use:

- optional image-only mechanism-of-action classifier
- CNN or ViT microscopy image experiment

## Molecule + Image Dataset

Path:

```text
../multimodal_datapipeline/data/processed/baseline_5_molecule_image.csv
```

Rows:

```text
300
```

Unique compounds:

```text
6
```

Unique MoA classes:

```text
3
```

Recommended use:

- small proof-of-concept molecule-image fusion experiment

Limitation:

- too small and imbalanced for a strong final claim

## Full Multimodal Dataset Status

Path:

```text
../multimodal_datapipeline/data/processed/baseline_7_molecule_protein_image_summary.json
```

Current status:

```text
blocked
```

Reason:

```text
0 exact SMILES overlap between molecule-protein and molecule-image processed tables
```

Implication:

The full molecule + protein + image model should be treated as future work unless additional curation aligns BBBC021 compounds to target-protein annotations.

