#!/usr/bin/env python3
"""Validate Baseline 6 readiness (aligned table + image embeddings)."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REQUIRED = [
    PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image.csv",
    PROJECT_ROOT / "data" / "interim" / "bbbc021_dinov2_embeddings.npz",
    PROJECT_ROOT / "data" / "interim" / "bbbc021_image_metadata.csv",
]


def main() -> None:
    missing = [str(path) for path in REQUIRED if not path.exists()]
    if missing:
        raise SystemExit("Baseline 6 is not ready; missing:\n  " + "\n  ".join(missing))
    print("Baseline 6 inputs present.")


if __name__ == "__main__":
    main()
