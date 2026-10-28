#!/usr/bin/env python3
"""Create the Davis benchmark table in the Baseline 4 format.

Input: ``python data/acquire_data.py davis`` (pairs, UniProt mapping, AlphaFold).
Output: data/processed/baseline_4_davis.csv with the same columns as the ChEMBL
table, so every Baseline 4 option works unchanged:

    target_chembl_id   Davis construct name (e.g. ``ABL1(T315I)``): one protein key per construct
    uniprot_id         mapped wild-type UniProt accession (cold-target splits group by this)
    protein_sequence   Davis construct sequence (carries point mutations)
    median_pchembl     pKd = -log10(Kd / 1e9)
    label              pKd >= --activity-threshold (7.0 is the usual Davis cut-off)

Constructs without a reviewed UniProt match or AlphaFold model are dropped and counted.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DAVIS = PROJECT_ROOT / "dataset_pipeline_output" / "davis"
ALPHAFOLD_META = PROJECT_ROOT / "dataset_pipeline_output" / "alphafold" / "metadata.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "baseline_4_davis.csv"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_4_davis_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the Davis table for Baseline 4.")
    parser.add_argument("--activity-threshold", type=float, default=7.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    pairs_csv = DAVIS / "davis_pairs.csv"
    mapping_csv = DAVIS / "davis_target_uniprot.csv"
    if not pairs_csv.exists() or not mapping_csv.exists():
        raise SystemExit("Missing Davis files. Run: python data/acquire_data.py davis")

    mapping = {row["davis_target"]: row for row in read_csv(mapping_csv)}
    # metadata.csv paths were written relative to different roots over time; locate by file name.
    structure_dir = ALPHAFOLD_META.parent / "structures"
    pdb_by_uniprot = {
        row["uniprot_id"]: str((structure_dir / Path(row["local_pdb_path"]).name).relative_to(PROJECT_ROOT))
        for row in read_csv(ALPHAFOLD_META)
        if row.get("status") == "downloaded"
        and row.get("local_pdb_path")
        and (structure_dir / Path(row["local_pdb_path"]).name).exists()
    }

    canonical: dict[str, str] = {}
    rows, dropped = [], Counter()
    for pair in read_csv(pairs_csv):
        target = mapping.get(pair["davis_target"], {})
        uniprot_id = target.get("uniprot_id", "")
        if target.get("status") != "mapped":
            dropped["target_not_mapped_to_uniprot"] += 1
            continue
        if uniprot_id not in pdb_by_uniprot:
            dropped["no_alphafold_structure"] += 1
            continue
        smiles = pair["smiles"]
        if smiles not in canonical:
            mol = Chem.MolFromSmiles(smiles)
            canonical[smiles] = Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""
        if not canonical[smiles]:
            dropped["invalid_smiles"] += 1
            continue
        pkd = float(pair["pkd"])
        rows.append(
            {
                "curated_smiles": canonical[smiles],
                "molecule_chembl_ids": f"PUBCHEM:{pair['drug_id']}",
                "target_chembl_id": pair["davis_target"],
                "uniprot_id": uniprot_id,
                "gene": target.get("gene", ""),
                "protein_sequence": pair["protein_sequence"],
                "alphafold_pdb_path": pdb_by_uniprot[uniprot_id],
                "kd_nm": pair["kd_nm"],
                "median_pchembl": f"{pkd:.4f}",
                "label": int(pkd >= args.activity_threshold),
            }
        )
    if not rows:
        raise SystemExit("No Davis rows survived mapping.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "output": str(args.output),
        "activity_threshold_pkd": args.activity_threshold,
        "rows": len(rows),
        "active_rows": sum(row["label"] == 1 for row in rows),
        "drugs": len({row["curated_smiles"] for row in rows}),
        "constructs": len({row["target_chembl_id"] for row in rows}),
        "uniprot_ids": len({row["uniprot_id"] for row in rows}),
        "dropped_pairs": dict(dropped),
        "unmapped_constructs": sorted(name for name, row in mapping.items() if row["status"] != "mapped"),
    }
    with args.summary.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "unmapped_constructs"}, indent=2))
    print(f"Unmapped constructs: {len(summary['unmapped_constructs'])}")


if __name__ == "__main__":
    main()
