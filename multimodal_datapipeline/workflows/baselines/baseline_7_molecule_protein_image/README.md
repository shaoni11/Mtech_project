# Baseline 7: Molecule + Protein + Image

Flow:

```text
SMILES -> molecule encoder (fingerprint / GNN) ------\
protein -> protein encoder (ESM-2 / AlphaFold GNN) ---> fusion -> active / inactive
compound images -> DINOv2 phenotype profile --------/      (+ optional MoA head)
```

Alignment (built by `curate_data.py`):

```text
BBBC021 compound -> InChIKey -> ChEMBL molecule -> human single-protein binding
activities (pChEMBL) -> UniProt -> AlphaFold structure; images -> phenotype profile
```

Data preparation:

```bash
python data/acquire_data.py bbbc021-chembl                                    # ChEMBL links + AlphaFold
python workflows/baselines/baseline_3_image_only/extract_image_embeddings.py   # all 13,200 images
python workflows/baselines/baseline_7_molecule_protein_image/curate_data.py
python workflows/baselines/baseline_7_molecule_protein_image/validate_inputs.py
```

Training options:

```text
--modalities molecule,protein,image   any subset (ablation); Baseline 6 = protein,image
--fusion concat|gated|cross_attention gated reports modality gates; cross_attention needs gnn + structure_gnn
--split compound|scaffold|cold_target|random   default compound: test compounds unseen in training
--seeds 0,1,2,3,4                     repeated splits; metrics.json reports mean and std
--init-from <b4 model.pt>             transfer: initialise encoders from Baseline 4 (ChEMBL / Davis)
--late-fusion-from <b4 model.pt>      late fusion: freeze Baseline 4, learn an image correction
--aux-moa-weight 0.5                  multitask: auxiliary MoA head on molecule/image embeddings
--modality-dropout 0.3                modality masking during training (gated fusion)
```

Examples:

```bash
python workflows/baselines/baseline_7_molecule_protein_image/train.py --out-dir results/baseline_7/full_gated
python workflows/baselines/baseline_7_molecule_protein_image/train.py --modalities molecule,protein --out-dir results/baseline_7/no_image
python workflows/baselines/baseline_7_molecule_protein_image/train.py --late-fusion-from results/baseline_4_fp_esm2_scaffold/model.pt --out-dir results/baseline_7/late_fusion
```

The aligned set is small (hundreds of pairs, ~60 compounds), so compare
configurations by the mean ± std over seeds, not a single run.
