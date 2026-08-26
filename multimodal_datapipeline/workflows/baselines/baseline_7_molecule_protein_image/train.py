#!/usr/bin/env python3
"""Guarded training entrypoint for Baseline 7."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image.csv"


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Baseline 7 is blocked until full molecule-protein-image alignment exists.")
    raise SystemExit("Next step: train MoleculeEncoder + ProteinEncoder + ImageEncoder + FusionHead.")


if __name__ == "__main__":
    main()

