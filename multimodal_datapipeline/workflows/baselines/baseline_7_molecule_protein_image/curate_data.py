#!/usr/bin/env python3
"""Check whether full molecule + protein + image alignment is available."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOLECULE_PROTEIN = PROJECT_ROOT / "data" / "processed" / "baseline_4_molecule_protein.csv"
MOLECULE_IMAGE = PROJECT_ROOT / "data" / "processed" / "baseline_5_molecule_image.csv"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image_summary.json"


def smiles_set(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {row[column] for row in csv.DictReader(handle) if row.get(column)}


def main() -> None:
    mp_smiles = smiles_set(MOLECULE_PROTEIN, "curated_smiles")
    mi_smiles = smiles_set(MOLECULE_IMAGE, "smiles")
    overlap = mp_smiles.intersection(mi_smiles)
    summary = {
        "status": "ready_for_alignment" if overlap else "blocked",
        "molecule_protein_smiles": len(mp_smiles),
        "molecule_image_smiles": len(mi_smiles),
        "exact_smiles_overlap": len(overlap),
        "reason": (
            "Exact SMILES overlap exists; create the aligned row-level table next."
            if overlap
            else "No exact SMILES overlap between molecule-protein and molecule-image processed tables."
        ),
        "required_output": "data/processed/baseline_7_molecule_protein_image.csv",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

