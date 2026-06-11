#!/usr/bin/env python3
"""Create the molecule + image processed table from BBBC021."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGES = PROJECT_ROOT / "data" / "processed" / "baseline_3_image_only.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "baseline_5_molecule_image.csv"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_5_molecule_image_summary.json"


def main() -> None:
    if not IMAGES.exists():
        raise SystemExit("Run baseline_3_image_only/curate_data.py first.")
    with IMAGES.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("smiles")]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "output": str(OUTPUT),
        "rows": len(rows),
        "unique_compounds": len({row["compound"] for row in rows}),
        "unique_moa": len({row["moa"] for row in rows}),
    }
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

