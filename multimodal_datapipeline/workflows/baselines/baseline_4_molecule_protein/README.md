# Baseline 4: Molecule + Protein

Flow:

```text
SMILES -> molecule encoder --------\
                                    -> fusion -> prediction
protein sequence -> protein encoder /
```

Current model components:

```text
package/multimodal_datapipeline/models/molecule_encoder.py
package/multimodal_datapipeline/models/protein_encoder.py
package/multimodal_datapipeline/models/fusion.py
```

Encoder options:

```text
--molecule-encoder fingerprint    Morgan fingerprint -> MLP (default)
--molecule-encoder gnn            atom/bond graph -> message-passing GNN
--protein-encoder esm2            frozen ESM-2 mean embedding -> trainable projection (default)
--protein-encoder structure_gnn   AlphaFold residue contact graph (C-alpha < 8 A, pLDDT node feature) -> GNN
--protein-encoder none            molecule only on the same rows and split (protein ablation)
```

Fusion options:

```text
--fusion concat            concatenation MLP (default)
--fusion gated             learned per-example modality gates (mean gates reported in metrics.json)
--fusion cross_attention   molecule atoms attend over AlphaFold residues (gnn + structure_gnn)
```

Training restores the epoch with the best validation ROC-AUC; `--patience N` adds early stopping.

Davis benchmark (442 kinase constructs, 68 inhibitors):

```bash
python data/acquire_data.py davis
python workflows/baselines/baseline_4_molecule_protein/curate_davis.py
python workflows/baselines/baseline_4_molecule_protein/train.py --data data/processed/baseline_4_davis.csv --split cold_target --out-dir results/baseline_4_davis_cold_target
```

On Davis each construct (including point mutants) is its own protein key; the
structure GNN uses the wild-type AlphaFold model with mutated residue identities,
and `cold_target` holds out whole genes (UniProt IDs).

Split options:

```text
--split random       stratified random rows (default; optimistic)
--split scaffold     Bemis-Murcko scaffold groups; no test scaffold seen in training
--split cold_target  whole proteins (UniProt) held out; meaningful on Davis, indicative only on the 12 ChEMBL kinases
```

Input table:

```text
curated_smiles
target_chembl_id
uniprot_id
protein_sequence
median_pchembl
label
```

Commands:

```bash
python workflows/baselines/baseline_2_protein_only/curate_data.py
python workflows/baselines/baseline_4_molecule_protein/curate_data.py
python workflows/baselines/baseline_4_molecule_protein/validate_inputs.py
python workflows/baselines/baseline_4_molecule_protein/train.py
```

Example runs:

```bash
python workflows/baselines/baseline_4_molecule_protein/train.py --split scaffold --out-dir results/baseline_4_fp_esm2_scaffold
python workflows/baselines/baseline_4_molecule_protein/train.py --split scaffold --molecule-encoder gnn --protein-encoder structure_gnn --out-dir results/baseline_4_gnn_structure_scaffold
python workflows/baselines/baseline_4_molecule_protein/train.py --max-examples 3000 --epochs 2   # smoke test
```
