#!/usr/bin/env python3
"""Curate ChEMBL molecule activity rows for molecule-only and DTI baselines.

The script reads ChEMBL activity rows, standardizes molecules with RDKit,
filters non-useful chemical records, aggregates repeated molecule-target
measurements, and writes a processed training table.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "dataset_pipeline_output" / "chembl" / "activities_multitarget.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "chembl_molecule_curated.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "processed" / "chembl_molecule_curated_summary.json"


METALS = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Al",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ag",
    "Cd",
    "Hg",
    "Pt",
    "Au",
    "Gd",
}

ORGANIC_ATOMS = {"C"}


def require_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for molecule curation. Install with "
            "`conda install -c conda-forge rdkit` or `python -m pip install rdkit`."
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    return Chem, Crippen, Descriptors, Lipinski, rdMolDescriptors


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def is_exact_ic50_nm(row: dict[str, str]) -> bool:
    return (
        row.get("standard_type") == "IC50"
        and row.get("standard_units") == "nM"
        and row.get("standard_relation") == "="
        and parse_float(row.get("pchembl_value")) is not None
    )


def largest_fragment(mol):
    Chem, *_ = require_rdkit()
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not fragments:
        return None
    return max(fragments, key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()))


def has_metal(mol) -> bool:
    return any(atom.GetSymbol() in METALS for atom in mol.GetAtoms())


def has_carbon(mol) -> bool:
    return any(atom.GetSymbol() in ORGANIC_ATOMS for atom in mol.GetAtoms())


def standardize_smiles(smiles: str) -> dict[str, object]:
    Chem, *_ = require_rdkit()
    result = {
        "status": "ok",
        "error": "",
        "curated_smiles": "",
        "fragment_removed": False,
    }

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        result["status"] = "invalid_smiles"
        result["error"] = "RDKit could not parse SMILES"
        return result

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        result["status"] = "sanitize_failed"
        result["error"] = str(exc)
        return result

    fragment = largest_fragment(mol)
    if fragment is None:
        result["status"] = "no_fragment"
        result["error"] = "No molecule fragment found"
        return result

    original_fragments = len(Chem.GetMolFrags(mol))
    result["fragment_removed"] = original_fragments > 1
    result["curated_smiles"] = Chem.MolToSmiles(fragment, isomericSmiles=True, canonical=True)
    return result


def molecule_properties(smiles: str) -> dict[str, float | int | bool]:
    Chem, Crippen, Descriptors, Lipinski, rdMolDescriptors = require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot compute properties for invalid SMILES: {smiles}")

    return {
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "has_metal": has_metal(mol),
        "has_carbon": has_carbon(mol),
    }


def passes_filters(props: dict[str, float | int | bool], args: argparse.Namespace) -> tuple[bool, str]:
    if args.remove_metals and props["has_metal"]:
        return False, "metal_containing"
    if args.require_carbon and not props["has_carbon"]:
        return False, "non_organic"
    if props["heavy_atom_count"] < args.min_heavy_atoms:
        return False, "too_few_heavy_atoms"
    if props["heavy_atom_count"] > args.max_heavy_atoms:
        return False, "too_many_heavy_atoms"
    if props["molecular_weight"] > args.max_molecular_weight:
        return False, "molecular_weight_high"
    if props["logp"] > args.max_logp:
        return False, "logp_high"
    if props["hbd"] > args.max_hbd:
        return False, "hbd_high"
    if props["hba"] > args.max_hba:
        return False, "hba_high"
    return True, "kept"


def read_and_filter_rows(args: argparse.Namespace) -> tuple[list[dict[str, object]], Counter]:
    counts = Counter()
    curated_rows = []

    with args.input.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "molecule_chembl_id",
            "target_chembl_id",
            "canonical_smiles",
            "standard_type",
            "standard_units",
            "standard_relation",
            "pchembl_value",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

        for row in reader:
            counts["raw_rows"] += 1
            if args.exact_ic50_nm_only and not is_exact_ic50_nm(row):
                counts["dropped_activity_quality"] += 1
                continue

            smiles = (row.get("canonical_smiles") or "").strip()
            pchembl = parse_float(row.get("pchembl_value"))
            if not smiles or pchembl is None:
                counts["dropped_missing_smiles_or_pchembl"] += 1
                continue

            standardized = standardize_smiles(smiles)
            if standardized["status"] != "ok":
                counts[f"dropped_{standardized['status']}"] += 1
                continue

            props = molecule_properties(str(standardized["curated_smiles"]))
            keep, reason = passes_filters(props, args)
            if not keep:
                counts[f"dropped_{reason}"] += 1
                continue

            counts["kept_activity_rows"] += 1
            if standardized["fragment_removed"]:
                counts["rows_with_fragment_removed"] += 1

            curated_rows.append(
                {
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "target_chembl_id": row["target_chembl_id"],
                    "original_smiles": smiles,
                    "curated_smiles": standardized["curated_smiles"],
                    "pchembl_value": pchembl,
                    "standard_type": row.get("standard_type", ""),
                    "standard_units": row.get("standard_units", ""),
                    "standard_relation": row.get("standard_relation", ""),
                    "fragment_removed": standardized["fragment_removed"],
                    **props,
                }
            )

    return curated_rows, counts


def aggregate_rows(rows: list[dict[str, object]], activity_threshold: float) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["curated_smiles"], row["target_chembl_id"])].append(row)

    output_rows = []
    for (curated_smiles, target_chembl_id), group in grouped.items():
        pchembl_values = [float(row["pchembl_value"]) for row in group]
        first = group[0]
        mean_pchembl = sum(pchembl_values) / len(pchembl_values)
        median_pchembl = statistics.median(pchembl_values)

        output_rows.append(
            {
                "target_chembl_id": target_chembl_id,
                "curated_smiles": curated_smiles,
                "molecule_chembl_ids": ";".join(sorted({str(row["molecule_chembl_id"]) for row in group})),
                "example_original_smiles": first["original_smiles"],
                "mean_pchembl": f"{mean_pchembl:.4f}",
                "median_pchembl": f"{median_pchembl:.4f}",
                "n_activity_rows": len(group),
                "label": int(median_pchembl >= activity_threshold),
                "heavy_atom_count": first["heavy_atom_count"],
                "molecular_weight": f"{float(first['molecular_weight']):.4f}",
                "logp": f"{float(first['logp']):.4f}",
                "hbd": first["hbd"],
                "hba": first["hba"],
                "rotatable_bonds": first["rotatable_bonds"],
                "tpsa": f"{float(first['tpsa']):.4f}",
                "fragment_removed": any(bool(row["fragment_removed"]) for row in group),
            }
        )

    output_rows.sort(key=lambda row: (row["target_chembl_id"], row["curated_smiles"]))
    return output_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_chembl_id",
        "curated_smiles",
        "molecule_chembl_ids",
        "example_original_smiles",
        "mean_pchembl",
        "median_pchembl",
        "n_activity_rows",
        "label",
        "heavy_atom_count",
        "molecular_weight",
        "logp",
        "hbd",
        "hba",
        "rotatable_bonds",
        "tpsa",
        "fragment_removed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curate ChEMBL molecule activity data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument("--min-heavy-atoms", type=int, default=6)
    parser.add_argument("--max-heavy-atoms", type=int, default=100)
    parser.add_argument("--max-molecular-weight", type=float, default=900.0)
    parser.add_argument("--max-logp", type=float, default=8.0)
    parser.add_argument("--max-hbd", type=int, default=10)
    parser.add_argument("--max-hba", type=int, default=15)
    parser.add_argument("--exact-ic50-nm-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--remove-metals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-carbon", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    curated_activity_rows, counts = read_and_filter_rows(args)
    aggregated_rows = aggregate_rows(curated_activity_rows, args.activity_threshold)

    write_csv(args.output, aggregated_rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "activity_threshold": args.activity_threshold,
        "filters": {
            "exact_ic50_nm_only": args.exact_ic50_nm_only,
            "min_heavy_atoms": args.min_heavy_atoms,
            "max_heavy_atoms": args.max_heavy_atoms,
            "max_molecular_weight": args.max_molecular_weight,
            "max_logp": args.max_logp,
            "max_hbd": args.max_hbd,
            "max_hba": args.max_hba,
            "remove_metals": args.remove_metals,
            "require_carbon": args.require_carbon,
        },
        "counts": dict(counts),
        "aggregated_molecule_target_rows": len(aggregated_rows),
        "unique_curated_smiles": len({row["curated_smiles"] for row in aggregated_rows}),
        "unique_targets": len({row["target_chembl_id"] for row in aggregated_rows}),
        "active_rows": sum(int(row["label"]) == 1 for row in aggregated_rows),
        "inactive_rows": sum(int(row["label"]) == 0 for row in aggregated_rows),
    }
    with args.summary.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("ChEMBL molecule curation complete")
    print(f"Raw rows: {counts['raw_rows']}")
    print(f"Kept activity rows: {counts['kept_activity_rows']}")
    print(f"Aggregated molecule-target rows: {len(aggregated_rows)}")
    print(f"Unique curated SMILES: {summary['unique_curated_smiles']}")
    print(f"Active/inactive rows: {summary['active_rows']}/{summary['inactive_rows']}")
    print(f"Wrote curated CSV: {args.output}")
    print(f"Wrote summary: {args.summary}")


if __name__ == "__main__":
    main()
