#!/usr/bin/env python3
"""Create the BBBC021 image-only processed table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the BBBC021 image-only processed table.")
    parser.add_argument("--bbbc-dir", type=Path, default=BBBC)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument(
        "--weeks",
        nargs="*",
        default=[],
        help="Optional week filters such as Week1 Week2. Defaults to all locally available weeks.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Fail if fewer than this many rows are produced.",
    )
    return parser


def row_week(row: dict[str, str]) -> str:
    plate = row.get("Image_Metadata_Plate_DAPI", "")
    return plate.split("_", 1)[0] if "_" in plate else plate


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "compound",
        "smiles",
        "concentration",
        "moa",
        "plate",
        "well",
        "replicate",
        "dapi_path",
        "tubulin_path",
        "actin_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    image_csv = args.bbbc_dir / "BBBC021_v1_image.csv"
    compound_csv = args.bbbc_dir / "BBBC021_v1_compound.csv"
    moa_csv = args.bbbc_dir / "BBBC021_v1_moa.csv"

    compounds = {row["compound"]: row.get("smiles", "") for row in read_csv(compound_csv)}
    moa = {
        (row["compound"], row["concentration"]): row["moa"]
        for row in read_csv(moa_csv)
    }
    files = {path.name: str(path) for path in args.image_root.rglob("*.tif")}
    week_filter = set(args.weeks)

    rows = []
    missing_files = 0
    missing_moa = 0
    skipped_week = 0
    available_moa_rows = 0
    missing_files_by_week = Counter()
    missing_moa_by_week = Counter()
    for row in read_csv(image_csv):
        week = row_week(row)
        if week_filter and week not in week_filter:
            skipped_week += 1
            continue

        compound = row["Image_Metadata_Compound"]
        concentration = row["Image_Metadata_Concentration"]
        label = moa.get((compound, concentration))
        if label is None:
            missing_moa += 1
            missing_moa_by_week[week] += 1
            continue
        available_moa_rows += 1

        dapi = files.get(row["Image_FileName_DAPI"])
        tubulin = files.get(row["Image_FileName_Tubulin"])
        actin = files.get(row["Image_FileName_Actin"])
        if not (dapi and tubulin and actin):
            missing_files += 1
            missing_files_by_week[week] += 1
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

    write_rows(args.output, rows)

    if len(rows) < args.min_rows:
        raise SystemExit(
            f"Only produced {len(rows)} rows, below --min-rows={args.min_rows}. "
            "Download/extract more BBBC021 images or relax filters."
        )

    class_counts = Counter(row["moa"] for row in rows)
    compound_counts = Counter(row["compound"] for row in rows)
    week_counts = Counter(row["plate"].split("_", 1)[0] for row in rows)
    plate_counts = Counter(row["plate"] for row in rows)

    summary = {
        "output": str(args.output),
        "bbbc_dir": str(args.bbbc_dir),
        "image_root": str(args.image_root),
        "weeks_filter": sorted(week_filter),
        "rows": len(rows),
        "unique_compounds": len({row["compound"] for row in rows}),
        "unique_moa": len({row["moa"] for row in rows}),
        "available_moa_rows": available_moa_rows,
        "missing_image_rows": missing_files,
        "missing_moa_rows": missing_moa,
        "skipped_week_rows": skipped_week,
        "class_counts": dict(sorted(class_counts.items())),
        "compound_counts": dict(sorted(compound_counts.items())),
        "week_counts": dict(sorted(week_counts.items())),
        "plate_counts": dict(sorted(plate_counts.items())),
        "missing_image_rows_by_week": dict(sorted(missing_files_by_week.items())),
        "missing_moa_rows_by_week": dict(sorted(missing_moa_by_week.items())),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
