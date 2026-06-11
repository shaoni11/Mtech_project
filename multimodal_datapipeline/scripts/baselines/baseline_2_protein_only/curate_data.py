#!/usr/bin/env python3
"""Create the protein-only processed table from ChEMBL targets and AlphaFold PDBs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_MAPPING = PROJECT_ROOT / "dataset_pipeline_output" / "chembl" / "target_mapping.csv"
ALPHAFOLD_META = PROJECT_ROOT / "dataset_pipeline_output" / "alphafold" / "metadata.csv"
CURATED_CHEMBL = PROJECT_ROOT / "data" / "processed" / "chembl_molecule_curated.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "baseline_2_protein_only.csv"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_2_protein_only_summary.json"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    for root in [PROJECT_ROOT, PROJECT_ROOT.parent, PROJECT_ROOT.parent.parent]:
        candidate = root / path
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / path


def sequence_from_pdb(path: Path) -> str:
    residues = []
    seen = set()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            chain = line[21].strip()
            resseq = line[22:27].strip()
            key = (chain, resseq)
            if key in seen:
                continue
            seen.add(key)
            residues.append(AA3_TO_1.get(line[17:20].strip(), "X"))
    return "".join(residues)


def main() -> None:
    primary_targets = {
        row["target_chembl_id"]: row
        for row in read_csv(TARGET_MAPPING)
        if row.get("target_type") == "SINGLE PROTEIN"
    }
    af_by_uniprot = {row["uniprot_id"]: row for row in read_csv(ALPHAFOLD_META)}

    labels = defaultdict(list)
    for row in read_csv(CURATED_CHEMBL):
        labels[row["target_chembl_id"]].append(int(row["label"]))

    rows = []
    for target_id, target in sorted(primary_targets.items()):
        uniprot_id = target["uniprot_id"]
        af_row = af_by_uniprot.get(uniprot_id)
        if not af_row:
            continue
        pdb_path = resolve_path(af_row["local_pdb_path"])
        sequence = sequence_from_pdb(pdb_path) if pdb_path.exists() else ""
        target_labels = labels.get(target_id, [])
        rows.append(
            {
                "target_chembl_id": target_id,
                "uniprot_id": uniprot_id,
                "pref_name": target.get("pref_name", ""),
                "organism": target.get("organism", ""),
                "alphafold_pdb_path": str(pdb_path),
                "protein_sequence": sequence,
                "sequence_length": len(sequence),
                "n_molecule_target_rows": len(target_labels),
                "active_fraction": f"{sum(target_labels) / len(target_labels):.6f}" if target_labels else "",
                "label": int(sum(target_labels) / len(target_labels) >= 0.5) if target_labels else "",
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
        "rows_with_sequence": sum(bool(row["protein_sequence"]) for row in rows),
        "rows_with_activity_labels": sum(bool(row["n_molecule_target_rows"]) for row in rows),
    }
    with SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
