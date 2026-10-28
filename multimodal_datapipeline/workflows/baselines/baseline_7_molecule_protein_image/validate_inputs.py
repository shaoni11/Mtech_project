#!/usr/bin/env python3
"""Validate Baseline 7 readiness and the aligned table contents."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image.csv"
EMBEDDINGS = PROJECT_ROOT / "data" / "interim" / "bbbc021_dinov2_embeddings.npz"
METADATA = PROJECT_ROOT / "data" / "interim" / "bbbc021_image_metadata.csv"


def main() -> None:
    missing = [str(path) for path in (DATA, EMBEDDINGS, METADATA) if not path.exists()]
    if missing:
        raise SystemExit("Baseline 7 is not ready; missing:\n  " + "\n  ".join(missing))
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with METADATA.open(newline="") as handle:
        imaged = {row["compound"] for row in csv.DictReader(handle)}
    summary = {
        "rows": len(rows),
        "missing_smiles": sum(not row["curated_smiles"] for row in rows),
        "missing_sequences": sum(not row["protein_sequence"] for row in rows),
        "missing_pdb_files": sum(
            not (Path(row["alphafold_pdb_path"]) if Path(row["alphafold_pdb_path"]).is_absolute()
                 else PROJECT_ROOT / row["alphafold_pdb_path"]).exists()
            for row in rows
        ),
        "compounds_without_images": sorted({row["compound"] for row in rows} - imaged),
    }
    print(json.dumps(summary, indent=2))
    if summary["missing_smiles"] or summary["missing_sequences"] or summary["missing_pdb_files"] or summary["compounds_without_images"]:
        raise SystemExit("Baseline 7 input validation failed.")


if __name__ == "__main__":
    main()
