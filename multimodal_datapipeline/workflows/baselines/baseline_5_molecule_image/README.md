# Baseline 5: Molecule + Image

Flow:

```text
SMILES -> Morgan fingerprint ------\
                                    -> gated fusion -> MoA class
cell images -> DINOv2 profile -----/
```

Unit of prediction: a treatment (compound, concentration). Its phenotype is the
plate-wise DMSO-centred mean DINOv2 embedding of its images. All modality
combinations use the same treatments (compounds with SMILES, DMSO excluded).

Evaluation: leave-one-compound-out (`--split loco`, the standard BBBC021 NSC
protocol) or one held-out-compound split (`--split compound`).

Commands:

```bash
python workflows/baselines/baseline_3_image_only/extract_image_embeddings.py   # once, ~3 min on MPS
python workflows/baselines/baseline_5_molecule_image/train.py --modalities image --model nn1 --out-dir results/baseline_5/b5_image_nn1
python workflows/baselines/baseline_5_molecule_image/train.py --modalities image --out-dir results/baseline_5/b5_image_only
python workflows/baselines/baseline_5_molecule_image/train.py --modalities molecule --out-dir results/baseline_5/b5_molecule_only
python workflows/baselines/baseline_5_molecule_image/train.py --modalities molecule,image --modality-dropout 0.5 --out-dir results/baseline_5/b5_molecule_image_mdrop
```

`--model nn1` is the classic 1-nearest-neighbour image baseline. `--modality-dropout`
randomly hides a modality during training (modality masking) so the fingerprint,
which memorises training compounds, cannot dominate the gate.

Outputs: `metrics.json` (treatment- and compound-level accuracy, macro-F1, mean
modality gates) and `test_predictions.csv` (per-fold predictions with gates).
