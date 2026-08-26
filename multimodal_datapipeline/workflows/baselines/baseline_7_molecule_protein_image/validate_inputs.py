#!/usr/bin/env python3
"""Validate Baseline 7 readiness."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image.csv"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(
            "Baseline 7 is not ready: missing full multimodal aligned table "
            "`data/processed/baseline_7_molecule_protein_image.csv`."
        )


if __name__ == "__main__":
    main()

