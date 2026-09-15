#!/usr/bin/env python3
"""Run a small Baseline 3 image-only experiment suite.

The suite compares lightweight CNN variants on the BBBC021 MoA classification
table and writes each run into its own results directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAIN_SCRIPT = Path(__file__).resolve().with_name("train.py")
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "baseline_3_image_only.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "baseline_3_image_only_experiments"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Baseline 3 image-only experiment suite.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-dinov2", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    return parser


def experiment_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    base = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--data",
        str(args.data),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.save_models:
        base.append("--save-model")
    if args.cache_images:
        base.append("--cache-images")

    experiments = [
        (
            "tiny_cnn",
            [
                *base,
                "--model",
                "tiny_cnn",
                "--out-dir",
                str(args.out_dir / "tiny_cnn"),
            ],
        ),
        (
            "small_cnn",
            [
                *base,
                "--model",
                "small_cnn",
                "--out-dir",
                str(args.out_dir / "small_cnn"),
            ],
        ),
        (
            "small_cnn_augmented",
            [
                *base,
                "--model",
                "small_cnn",
                "--augment",
                "--out-dir",
                str(args.out_dir / "small_cnn_augmented"),
            ],
        ),
    ]

    if args.include_dinov2:
        experiments.append(
            (
                "dinov2_linear",
                [
                    *base,
                    "--model",
                    "dinov2_linear",
                    "--out-dir",
                    str(args.out_dir / "dinov2_linear"),
                ],
            )
        )

    return experiments


def load_test_metrics(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open() as handle:
        metrics = json.load(handle)
    return metrics["test"]


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for name, command in experiment_commands(args):
        print(f"\n=== Running {name} ===", flush=True)
        subprocess.run(command, check=True)

        run_dir = args.out_dir / name
        test_metrics = load_test_metrics(run_dir)
        summary_rows.append({"experiment": name, **test_metrics})

    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary_rows, handle, indent=2)

    print("\nExperiment summary")
    print("experiment,accuracy,macro_precision,macro_recall,macro_f1,loss")
    for row in summary_rows:
        print(
            "{experiment},{accuracy:.4f},{macro_precision:.4f},"
            "{macro_recall:.4f},{macro_f1:.4f},{loss:.4f}".format(**row)
        )
    print(f"\nWrote summary: {summary_path}")


if __name__ == "__main__":
    main()
