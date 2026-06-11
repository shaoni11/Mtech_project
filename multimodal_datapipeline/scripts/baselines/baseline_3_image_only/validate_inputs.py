#!/usr/bin/env python3
"""Validate the Baseline 3 image-only processed table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "processed" / "baseline_3_image_only.csv"


def main() -> None:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = 0
    for row in rows:
        missing += sum(not Path(row[col]).exists() for col in ["dapi_path", "tubulin_path", "actin_path"])
    summary = {
        "input": str(DATA),
        "rows": len(rows),
        "unique_moa": len({row["moa"] for row in rows}),
        "missing_channel_files": missing,
        "rows_without_smiles": sum(not row["smiles"] for row in rows),
    }
    print(json.dumps(summary, indent=2))
    if missing:
        raise SystemExit("Image input validation failed.")


if __name__ == "__main__":
    main()

