# Baseline 3: Image Only

Flow:

```text
BBBC021 cell image -> image encoder -> prediction
```

Current model options:

```text
tiny_cnn         -> lightweight CNN sanity-check model
small_cnn        -> default custom microscopy CNN
dinov2_linear    -> frozen DINOv2 backbone + trainable classifier head
```

Planned input table:

```text
image_path
compound
moa_label
metadata fields
```

The default Baseline 3 experiment uses `small_cnn` because the current BBBC021
table is small. `dinov2_linear` is available as a transfer-learning variant when
the Hugging Face model weights are installed or cached locally.

Commands:

```bash
python workflows/baselines/baseline_3_image_only/curate_data.py
python workflows/baselines/baseline_3_image_only/validate_inputs.py
python workflows/baselines/baseline_3_image_only/train.py
```

Acquire more BBBC021 image data before curation:

```bash
python data/acquire_data.py --download-missing-phase1 --download-bbbc021-images --bbbc021-image-set all --bbbc021-extract
```

Download only selected additional weeks:

```bash
python data/acquire_data.py --download-missing-phase1 --download-bbbc021-images --bbbc021-weeks Week2 Week3 --bbbc021-extract
```

Download the BBBC021 CellProfiler analysis and illumination pipeline files:

```bash
python data/acquire_data.py --download-missing-phase1 --download-bbbc021-cellprofiler
```

Rebuild the processed table from all locally extracted weeks:

```bash
python workflows/baselines/baseline_3_image_only/curate_data.py
```

Rebuild only selected weeks:

```bash
python workflows/baselines/baseline_3_image_only/curate_data.py --weeks Week1 Week2
```

Run a single explicit model:

```bash
python workflows/baselines/baseline_3_image_only/train.py --model small_cnn
python workflows/baselines/baseline_3_image_only/train.py --model small_cnn --augment
python workflows/baselines/baseline_3_image_only/train.py --model dinov2_linear
```

Realistic splits (`--split`):

```text
random    stratified random images (default; optimistic)
compound  hold out whole compounds inside every MoA class (standard BBBC021 protocol)
week      hold out whole acquisition weeks (cross-batch robustness)
```

```bash
python workflows/baselines/baseline_3_image_only/train.py --split compound --exclude-dmso --out-dir results/baseline_3_compound_split
python workflows/baselines/baseline_3_image_only/train.py --split week --out-dir results/baseline_3_week_split
python workflows/baselines/baseline_3_image_only/run_experiments.py --split compound --exclude-dmso --include-dinov2 --out-dir results/baseline_3_compound_split_suite
```

DMSO is a single compound, so with `--split compound` and without `--exclude-dmso`
its images are split randomly; this is recorded as `split.row_split_classes`.
Macro metrics average over the classes present in a split.

Run the comparison suite:

```bash
python workflows/baselines/baseline_3_image_only/run_experiments.py
```

Optionally include DINOv2:

```bash
python workflows/baselines/baseline_3_image_only/run_experiments.py --include-dinov2
```
