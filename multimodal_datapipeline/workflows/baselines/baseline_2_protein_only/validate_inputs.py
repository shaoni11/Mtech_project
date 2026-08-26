#!/usr/bin/env python3
"""Validate the Baseline 2 protein-only processed table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_2_protein_only.csv"


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = {
        "input": str(DATA),
        "rows": len(rows),
        "missing_sequences": sum(not row["protein_sequence"] for row in rows),
        "missing_pdb_files": sum(not Path(row["alphafold_pdb_path"]).exists() for row in rows),
        "too_short_sequences": sum(0 < len(row["protein_sequence"]) < 30 for row in rows),
    }
    print(json.dumps(summary, indent=2))
    if summary["missing_sequences"] or summary["missing_pdb_files"]:
        raise SystemExit("Protein input validation failed.")


if __name__ == "__main__":
    main()

