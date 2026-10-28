#!/usr/bin/env python3
"""Create the aligned molecule + protein + image table (Baselines 6 and 7).

Alignment: BBBC021 compound -> ChEMBL molecule (InChIKey) -> human single-protein
binding activities -> UniProt -> AlphaFold structure. The image modality is the
compound's phenotype profile, joined at training time from the BBBC021 image
embeddings, so every row here has a compound with BBBC021 images.

Inputs come from ``python data/acquire_data.py bbbc021-chembl``.
One row per (compound, protein): median pChEMBL and an active label.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from multimodal_datapipeline.models.protein_structure_encoder import parse_alphafold_pdb  # noqa: E402


ACTIVITIES = PROJECT_ROOT / "dataset_pipeline_output" / "chembl" / "bbbc021_compound_activities.csv"
CHEMBL_MAP = PROJECT_ROOT / "dataset_pipeline_output" / "chembl" / "bbbc021_compound_chembl_map.csv"
ALPHAFOLD_META = PROJECT_ROOT / "dataset_pipeline_output" / "alphafold" / "metadata.csv"
IMAGE_METADATA = PROJECT_ROOT / "data" / "interim" / "bbbc021_image_metadata.csv"
MOA_CSV = PROJECT_ROOT / "dataset_pipeline_output" / "bbbc021" / "BBBC021_v1_moa.csv"
OUTPUT = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image.csv"
SUMMARY = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image_summary.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_smiles(smiles: str) -> str:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the aligned molecule + protein + image table.")
    parser.add_argument("--activity-threshold", type=float, default=6.0, help="Active if median pChEMBL >= this.")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    for path, hint in [
        (ACTIVITIES, "python data/acquire_data.py bbbc021-chembl"),
        (IMAGE_METADATA, "python workflows/baselines/baseline_3_image_only/extract_image_embeddings.py"),
    ]:
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run: {hint}")

    compounds_with_images = Counter(row["compound"] for row in read_csv(IMAGE_METADATA))
    chembl_ids = {row["compound"]: row.get("molecule_chembl_id", "") for row in read_csv(CHEMBL_MAP)}
    moa_by_compound = {row["compound"]: row["moa"] for row in read_csv(MOA_CSV)}
    # metadata.csv paths were written relative to different roots over time; locate by file name.
    structure_dir = ALPHAFOLD_META.parent / "structures"
    pdb_by_uniprot = {
        row["uniprot_id"]: str((structure_dir / Path(row["local_pdb_path"]).name).relative_to(PROJECT_ROOT))
        for row in read_csv(ALPHAFOLD_META)
        if row.get("status") == "downloaded"
        and row.get("local_pdb_path")
        and (structure_dir / Path(row["local_pdb_path"]).name).exists()
    }

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    dropped = Counter()
    for row in read_csv(ACTIVITIES):
        uniprot_ids = row.get("uniprot_ids", "")
        if row.get("target_type") != "SINGLE PROTEIN" or not uniprot_ids or ";" in uniprot_ids:
            dropped["not_single_protein"] += 1
            continue
        if not row.get("pchembl_value"):
            dropped["no_pchembl"] += 1
            continue
        if row["compound"] not in compounds_with_images:
            dropped["compound_without_images"] += 1
            continue
        if uniprot_ids not in pdb_by_uniprot:
            dropped["no_alphafold_structure"] += 1
            continue
        groups[(row["compound"], uniprot_ids)].append(row)

    sequences: dict[str, str] = {}
    smiles_cache: dict[str, str] = {}
    rows = []
    for (compound, uniprot_id), group in sorted(groups.items()):
        pdb_path = pdb_by_uniprot[uniprot_id]
        if uniprot_id not in sequences:
            resolved = Path(pdb_path) if Path(pdb_path).is_absolute() else PROJECT_ROOT / pdb_path
            sequences[uniprot_id] = parse_alphafold_pdb(resolved).sequence
        smiles = group[0]["bbbc021_smiles"]
        if smiles not in smiles_cache:
            smiles_cache[smiles] = canonical_smiles(smiles)
        values = [float(row["pchembl_value"]) for row in group]
        median = statistics.median(values)
        rows.append(
            {
                "compound": compound,
                "curated_smiles": smiles_cache[smiles],
                "molecule_chembl_ids": chembl_ids.get(compound, ""),
                "target_chembl_id": group[0]["target_chembl_id"],
                "target_pref_name": group[0]["target_pref_name"],
                "uniprot_id": uniprot_id,
                "protein_sequence": sequences[uniprot_id],
                "alphafold_pdb_path": pdb_path,
                "n_activity_rows": len(group),
                "median_pchembl": f"{median:.4f}",
                "label": int(median >= args.activity_threshold),
                "moa": moa_by_compound.get(compound, ""),
                "n_images": compounds_with_images[compound],
            }
        )
    rows = [row for row in rows if row["curated_smiles"]]
    if not rows:
        raise SystemExit("No aligned compound-protein-image rows were produced.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "ready",
        "output": str(args.output),
        "activity_threshold": args.activity_threshold,
        "rows": len(rows),
        "active_rows": sum(row["label"] == 1 for row in rows),
        "inactive_rows": sum(row["label"] == 0 for row in rows),
        "compounds": len({row["compound"] for row in rows}),
        "compounds_with_moa": len({row["compound"] for row in rows if row["moa"]}),
        "proteins": len({row["uniprot_id"] for row in rows}),
        "dropped_activity_rows": dict(dropped),
        "rows_per_compound": dict(Counter(row["compound"] for row in rows).most_common()),
    }
    with args.summary.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows_per_compound"}, indent=2))


if __name__ == "__main__":
    main()
