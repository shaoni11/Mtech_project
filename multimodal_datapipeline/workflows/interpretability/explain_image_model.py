#!/usr/bin/env python3
"""Grad-CAM explanations for a trained Baseline 3 microscopy CNN.

Needs a run trained with ``--save-model`` (tiny_cnn or small_cnn). For the most
confident correct test predictions per MoA class, writes a PNG with the
DAPI/tubulin/actin composite and the Grad-CAM heatmap of the predicted class.

Usage:
    python workflows/interpretability/explain_image_model.py --run-dir results/baseline_3_compound_split
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from multimodal_datapipeline.interpretability import GradCAM, microscopy_composite  # noqa: E402
from multimodal_datapipeline.utils.baseline_launcher import load_baseline_module  # noqa: E402
from multimodal_datapipeline.utils.microscopy import CHANNEL_COLUMNS, load_three_channel_array  # noqa: E402


B3 = load_baseline_module("baseline_3_image_only")


def last_conv(model: nn.Module) -> nn.Module:
    convs = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not convs:
        raise SystemExit("Grad-CAM needs a CNN (tiny_cnn or small_cnn); this model has no Conv2d layer.")
    return convs[-1]


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Grad-CAM for a Baseline 3 CNN.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, default=None)
    cli = parser.parse_args()
    out_dir = cli.out_dir or cli.run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = cli.run_dir / "model.pt"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found; train Baseline 3 with --save-model.")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    run_args = checkpoint["args"]
    label_to_id = checkpoint["label_to_id"]
    model = B3.build_model(
        model_name=run_args["model"],
        num_classes=len(label_to_id),
        dropout=run_args["dropout"],
        dinov2_model_name=run_args["dinov2_model_name"],
    )
    model.load_state_dict(checkpoint["model"])
    cam = GradCAM(model, last_conv(model))

    with (cli.run_dir / "test_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if CHANNEL_COLUMNS[0] not in predictions[0]:
        raise SystemExit("test_predictions.csv has no image paths; re-run training with the current train.py.")
    by_class = defaultdict(list)
    for prediction in predictions:
        if prediction["true_moa"] == prediction["predicted_moa"]:
            by_class[prediction["true_moa"]].append(prediction)

    written = []
    for moa, items in sorted(by_class.items()):
        items.sort(key=lambda p: float(p["confidence"]), reverse=True)
        for rank, prediction in enumerate(items[: cli.per_class], start=1):
            array = load_three_channel_array([prediction[c] for c in CHANNEL_COLUMNS], run_args["image_size"])
            heatmap, class_index = cam(torch.from_numpy(array), label_to_id[moa])
            figure, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(microscopy_composite(array))
            axes[0].set_title(f"{prediction['compound']} ({prediction['concentration']})", fontsize=9)
            axes[1].imshow(microscopy_composite(array))
            axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
            axes[1].set_title(f"Grad-CAM: {moa} ({float(prediction['confidence']):.2f})", fontsize=9)
            for axis in axes:
                axis.axis("off")
            name = f"{''.join(ch if ch.isalnum() else '_' for ch in moa)}_{rank}.png"
            figure.tight_layout()
            figure.savefig(out_dir / name, dpi=120)
            plt.close(figure)
            written.append({"moa": moa, "compound": prediction["compound"], "file": name})
    cam.close()
    with (out_dir / "gradcam_index.json").open("w") as handle:
        json.dump(written, handle, indent=2)
    print(f"Wrote {len(written)} Grad-CAM figures to {out_dir}")


if __name__ == "__main__":
    main()
