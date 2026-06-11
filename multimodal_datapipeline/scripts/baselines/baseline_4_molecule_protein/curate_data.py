#!/usr/bin/env python3
"""Create the molecule + protein DTI processed table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOLECULES = PROJECT_ROOT / "data" / "processed" / "chembl_molecule_curated.csv"
PROTEINS = PROJECT_ROOT / "data" / "processed" / "baseline_2_protein_only.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "baseline_4_molecule_protein.csv"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_4_molecule_protein_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if not PROTEINS.exists():
        raise SystemExit("Run baseline_2_protein_only/curate_data.py first.")
    proteins = {row["target_chembl_id"]: row for row in read_csv(PROTEINS)}
    rows = []
    skipped = 0
    for mol in read_csv(MOLECULES):
        protein = proteins.get(mol["target_chembl_id"])
        if not protein or not protein["protein_sequence"]:
            skipped += 1
            continue
        rows.append(
            {
                "curated_smiles": mol["curated_smiles"],
                "target_chembl_id": mol["target_chembl_id"],
                "uniprot_id": protein["uniprot_id"],
                "protein_sequence": protein["protein_sequence"],
                "alphafold_pdb_path": protein["alphafold_pdb_path"],
                "median_pchembl": mol["median_pchembl"],
                "label": mol["label"],
                "molecule_chembl_ids": mol["molecule_chembl_ids"],
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
        "skipped_without_protein": skipped,
        "unique_targets": len({row["target_chembl_id"] for row in rows}),
        "unique_molecules": len({row["curated_smiles"] for row in rows}),
    }
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

