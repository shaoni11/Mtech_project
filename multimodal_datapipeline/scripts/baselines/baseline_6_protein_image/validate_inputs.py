#!/usr/bin/env python3
"""Validate Baseline 6 readiness."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_6_protein_image.csv"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            "Baseline 6 is not ready: missing protein-image aligned table "
            "`data/processed/baseline_6_protein_image.csv`."
        )


if __name__ == "__main__":
    main()

