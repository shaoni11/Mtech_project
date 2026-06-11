#!/usr/bin/env python3
"""Validate the Baseline 4 molecule + protein table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_4_molecule_protein.csv"


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = {
        "input": str(DATA),
        "rows": len(rows),
        "missing_smiles": sum(not row["curated_smiles"] for row in rows),
        "missing_sequences": sum(not row["protein_sequence"] for row in rows),
        "missing_labels": sum(row["label"] == "" for row in rows),
    }
    print(json.dumps(summary, indent=2))
    if summary["missing_smiles"] or summary["missing_sequences"] or summary["missing_labels"]:
        raise SystemExit("Molecule + protein input validation failed.")


if __name__ == "__main__":
    main()

