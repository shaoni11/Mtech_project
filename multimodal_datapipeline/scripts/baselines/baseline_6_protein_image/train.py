#!/usr/bin/env python3
"""Guarded training entrypoint for Baseline 6."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_6_protein_image.csv"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Baseline 6 is blocked until protein-image alignment exists.")
    raise SystemExit("Next step: train ProteinEncoder + ImageEncoder + FusionHead.")


if __name__ == "__main__":
    main()

