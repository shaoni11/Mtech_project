#!/usr/bin/env python3
"""Create the BBBC021 image-only processed table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BBBC = PROJECT_ROOT / "dataset_pipeline_output" / "bbbc021"
IMAGE_CSV = BBBC / "BBBC021_v1_image.csv"
COMPOUND_CSV = BBBC / "BBBC021_v1_compound.csv"
MOA_CSV = BBBC / "BBBC021_v1_moa.csv"
IMAGE_ROOT = BBBC / "images"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "baseline_3_image_only.csv"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_3_image_only_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def image_index() -> dict[str, str]:
    return {path.name: str(path) for path in IMAGE_ROOT.rglob("*.tif")}


def main() -> None:
    compounds = {row["compound"]: row.get("smiles", "") for row in read_csv(COMPOUND_CSV)}
    moa = {
        (row["compound"], row["concentration"]): row["moa"]
        for row in read_csv(MOA_CSV)
    }
    files = image_index()

    rows = []
    missing_files = 0
    missing_moa = 0
    for row in read_csv(IMAGE_CSV):
        compound = row["Image_Metadata_Compound"]
        concentration = row["Image_Metadata_Concentration"]
        label = moa.get((compound, concentration))
        if label is None:
            missing_moa += 1
            continue
        dapi = files.get(row["Image_FileName_DAPI"])
        tubulin = files.get(row["Image_FileName_Tubulin"])
        actin = files.get(row["Image_FileName_Actin"])
        if not (dapi and tubulin and actin):
            missing_files += 1
            continue
        rows.append(
            {
                "compound": compound,
                "smiles": compounds.get(compound, ""),
                "concentration": concentration,
                "moa": label,
                "plate": row["Image_Metadata_Plate_DAPI"],
                "well": row["Image_Metadata_Well_DAPI"],
                "replicate": row["Replicate"],
                "dapi_path": dapi,
                "tubulin_path": tubulin,
                "actin_path": actin,
            }
        )

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
        "missing_image_rows": missing_files,
        "missing_moa_rows": missing_moa,
    }
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

