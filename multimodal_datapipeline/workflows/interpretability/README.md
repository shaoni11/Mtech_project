# Interpretability

`explain_activity_model.py` — molecule + protein models (Baseline 4, ChEMBL or Davis):

```bash
python workflows/interpretability/explain_activity_model.py --run-dir results/<baseline_4_run> --top-k 10
```

For the most confident correctly predicted actives it writes, under `<run>/explanations/`:

- `*_atoms.svg`: atoms coloured by gradient x input (red = pushes towards active, blue = against)
- `*_residues.pdb`: the AlphaFold model with B-factor = residue attribution (0-100);
  in PyMOL: `load file.pdb; spectrum b, blue_white_red`
- `*_residues.csv`: top residues, with cross-attention weights for `--fusion cross_attention`
- `explanations.csv`: one row per pair, including modality gates for `--fusion gated`

Residue-level output needs `--protein-encoder structure_gnn`.

`explain_image_model.py` — Grad-CAM for Baseline 3 CNNs trained with `--save-model`:

```bash
python workflows/baselines/baseline_3_image_only/train.py --model small_cnn --split compound --exclude-dmso --save-model --out-dir results/baseline_3_compound_split
python workflows/interpretability/explain_image_model.py --run-dir results/baseline_3_compound_split
```

Modality-level attribution is reported directly by the gated-fusion runs
(`mean_gate_*` in Baseline 4/5/7 `metrics.json`).
