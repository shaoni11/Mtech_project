#!/usr/bin/env python3
"""Guarded training entrypoint for Baseline 5."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_5_molecule_image.csv"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run curate_data.py first for Baseline 5.")
    raise SystemExit(
        "Baseline 5 data is prepared. Next step: train MoleculeEncoder + ImageEncoder + FusionHead."
    )


if __name__ == "__main__":
    main()

