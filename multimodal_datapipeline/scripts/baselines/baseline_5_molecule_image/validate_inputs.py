#!/usr/bin/env python3
"""Validate the Baseline 5 molecule + image table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_5_molecule_image.csv"


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing_files = sum(
        not Path(row[col]).exists()
        for row in rows
        for col in ["dapi_path", "tubulin_path", "actin_path"]
    )
    summary = {
        "input": str(DATA),
        "rows": len(rows),
        "missing_smiles": sum(not row["smiles"] for row in rows),
        "missing_channel_files": missing_files,
    }
    print(json.dumps(summary, indent=2))
    if summary["missing_smiles"] or summary["missing_channel_files"]:
        raise SystemExit("Molecule + image input validation failed.")


if __name__ == "__main__":
    main()

