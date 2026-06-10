#!/usr/bin/env python3
"""Validate ChEMBL SMILES before using the RDKit molecule encoder."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from multimodal_datapipeline.models.molecule_encoder import (  # noqa: E402
    MoleculeEncoder,
    MorganFingerprintConfig,
    MorganFingerprintFeaturizer,
)


DEFAULT_DATA = PROJECT_ROOT / "dataset_pipeline_output" / "chembl" / "activities_multitarget.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "experiments" / "molecule_encoder_validation"


def require_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdMolDescriptors
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for this validation. Install it first, preferably with "
            "`conda install -c conda-forge rdkit`, or try `python -m pip install rdkit`."
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    return Chem, rdMolDescriptors


def load_unique_smiles(csv_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows_by_smiles = {}
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"canonical_smiles", "molecule_chembl_id", "target_chembl_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

        for row in reader:
            smiles = (row.get("canonical_smiles") or "").strip()
            if not smiles or smiles in rows_by_smiles:
                continue
            rows_by_smiles[smiles] = {
                "canonical_smiles": smiles,
                "molecule_chembl_id": row.get("molecule_chembl_id", ""),
                "target_chembl_id": row.get("target_chembl_id", ""),
            }
            if limit is not None and len(rows_by_smiles) >= limit:
                break
    return list(rows_by_smiles.values())


def validate_one(smiles: str, featurizer: MorganFingerprintFeaturizer) -> dict[str, object]:
    Chem, rdMolDescriptors = require_rdkit()
    result = {
        "status": "ok",
        "error": "",
        "canonical_smiles_rdkit": "",
        "has_stereochemistry": False,
        "has_unassigned_atom_stereo": False,
        "has_unassigned_bond_stereo": False,
        "n_atoms": 0,
        "fingerprint_bits_on": 0,
    }

    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        result["status"] = "invalid_smiles"
        result["error"] = "RDKit could not parse SMILES"
        return result

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit exposes several sanitization exception types.
        result["status"] = "sanitize_failed"
        result["error"] = str(exc)
        return result

    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    unassigned_atoms, unassigned_bonds = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol), 0
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOANY:
            unassigned_bonds += 1

    try:
        fingerprint = featurizer.transform_one(smiles)
    except Exception as exc:
        result["status"] = "fingerprint_failed"
        result["error"] = str(exc)
        return result

    result.update(
        {
            "canonical_smiles_rdkit": Chem.MolToSmiles(mol, isomericSmiles=True),
            "has_stereochemistry": any(atom.HasProp("_CIPCode") for atom in mol.GetAtoms())
            or any(bond.GetStereo() != Chem.BondStereo.STEREONONE for bond in mol.GetBonds()),
            "has_unassigned_atom_stereo": unassigned_atoms > 0,
            "has_unassigned_bond_stereo": unassigned_bonds > 0,
            "n_atoms": mol.GetNumAtoms(),
            "fingerprint_bits_on": int(fingerprint.sum()),
        }
    )
    return result


def run_encoder_smoke_test(smiles_batch: list[str], embedding_dim: int) -> dict[str, object]:
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "status": "skipped",
            "reason": "PyTorch is not installed; fingerprint validation still ran.",
        }

    try:
        model = MoleculeEncoder(embedding_dim=embedding_dim)
        model.eval()
        with torch.no_grad():
            embeddings = model.forward_smiles(smiles_batch)
    except Exception as exc:
        return {"status": "failed", "reason": str(exc)}

    return {
        "status": "ok",
        "batch_size": len(smiles_batch),
        "embedding_shape": list(embeddings.shape),
        "all_finite": bool(torch.isfinite(embeddings).all().item()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate RDKit molecule encoding inputs.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional max unique SMILES to validate.")
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--smoke-test-size", type=int, default=16)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    featurizer = MorganFingerprintFeaturizer(
        MorganFingerprintConfig(radius=args.radius, n_bits=args.fingerprint_bits)
    )
    examples = load_unique_smiles(args.data, limit=args.limit)

    report_rows = []
    counts = Counter()
    valid_smiles = []
    for example in examples:
        validation = validate_one(example["canonical_smiles"], featurizer)
        counts[validation["status"]] += 1
        if validation["status"] == "ok":
            valid_smiles.append(example["canonical_smiles"])
        report_rows.append({**example, **validation})

    report_path = args.out_dir / "smiles_validation_report.csv"
    with report_path.open("w", newline="") as handle:
        fieldnames = [
            "molecule_chembl_id",
            "target_chembl_id",
            "canonical_smiles",
            "status",
            "error",
            "canonical_smiles_rdkit",
            "has_stereochemistry",
            "has_unassigned_atom_stereo",
            "has_unassigned_bond_stereo",
            "n_atoms",
            "fingerprint_bits_on",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    smoke_batch = valid_smiles[: args.smoke_test_size]
    smoke_test = (
        run_encoder_smoke_test(smoke_batch, args.embedding_dim)
        if smoke_batch
        else {"status": "skipped", "reason": "No valid SMILES available."}
    )

    summary = {
        "input_csv": str(args.data),
        "n_unique_smiles_checked": len(examples),
        "status_counts": dict(counts),
        "n_with_stereochemistry": sum(row["has_stereochemistry"] is True for row in report_rows),
        "n_with_unassigned_atom_stereo": sum(row["has_unassigned_atom_stereo"] is True for row in report_rows),
        "n_with_unassigned_bond_stereo": sum(row["has_unassigned_bond_stereo"] is True for row in report_rows),
        "encoder_smoke_test": smoke_test,
        "report_csv": str(report_path),
    }
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print("Molecule encoder validation complete")
    print(f"Checked unique SMILES: {len(examples)}")
    print(f"Status counts: {dict(counts)}")
    print(f"Unassigned atom stereo: {summary['n_with_unassigned_atom_stereo']}")
    print(f"Unassigned bond stereo: {summary['n_with_unassigned_bond_stereo']}")
    print(f"Encoder smoke test: {smoke_test['status']}")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
