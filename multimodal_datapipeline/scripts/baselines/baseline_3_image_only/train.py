#!/usr/bin/env python3
"""Guarded training entrypoint for Baseline 3."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_3_image_only.csv"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run curate_data.py first for Baseline 3.")
    raise SystemExit(
        "Baseline 3 data is prepared, but image model training is not implemented yet. "
        "Next step: load DAPI/Tubulin/Actin images and train DINOv2 embeddings for MoA prediction."
    )


if __name__ == "__main__":
    main()

