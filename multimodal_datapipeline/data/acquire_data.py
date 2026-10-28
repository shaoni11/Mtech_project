#!/usr/bin/env python3
"""Unified data acquisition entrypoint.

Default mode delegates to the existing Phase 1 dataset pipeline without
changing its CLI arguments. Subcommands:

    cytoskeleton-chembl  actin/tubulin ChEMBL acquisition
    davis                Davis kinase benchmark + UniProt mapping + AlphaFold structures
    bbbc021-chembl       ChEMBL binding activities for BBBC021 compounds + AlphaFold structures
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))


def run_dataset_pipeline() -> None:
    from data.pipelines.dataset_pipeline import main

    main()


def find_human_targets(target_client, target_spec: dict) -> list[dict]:
    """Find ChEMBL targets for an exact UniProt accession."""

    gene = target_spec["gene"]
    uniprot_id = target_spec["uniprot_id"]

    print(f"Searching ChEMBL target for {gene} ({uniprot_id})", flush=True)

    try:
        targets = list(target_client.filter(target_components__accession=uniprot_id))
    except Exception as exc:
        print(f"  UniProt lookup failed for {gene}: {exc}", flush=True)
        targets = []

    if not targets:
        print(f"  No exact UniProt-linked ChEMBL target found for {gene}; skipping", flush=True)
        return []

    human_targets = [
        target
        for target in targets
        if target.get("organism") == "Homo sapiens"
        and target.get("target_type") in {"SINGLE PROTEIN", "PROTEIN FAMILY", "PROTEIN COMPLEX"}
    ]

    seen = set()
    unique_targets = []
    for target in human_targets:
        target_chembl_id = target.get("target_chembl_id")
        if target_chembl_id and target_chembl_id not in seen:
            unique_targets.append(target)
            seen.add(target_chembl_id)

    print(f"  Found {len(unique_targets)} human ChEMBL target(s)", flush=True)
    return unique_targets


def run_cytoskeleton_chembl() -> None:
    """Download ChEMBL activity records for actin/tubulin bridge targets."""

    import pandas as pd
    from chembl_webresource_client.new_client import new_client

    targets_to_query = [
        {"gene": "ACTB", "uniprot_id": "P60709"},
        {"gene": "TUBA1A", "uniprot_id": "Q71U36"},
        {"gene": "TUBB", "uniprot_id": "P07437"},
        {"gene": "TUBB3", "uniprot_id": "Q13509"},
    ]
    standard_types = ["IC50", "Ki", "Kd", "EC50"]

    out_dir = PROJECT_ROOT / "dataset_pipeline_output" / "chembl"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cytoskeleton_activities.csv"

    target_client = new_client.target
    activity_client = new_client.activity
    rows = []

    for target_spec in targets_to_query:
        gene = target_spec["gene"]
        uniprot_id = target_spec["uniprot_id"]
        targets = find_human_targets(target_client, target_spec)

        for target in targets:
            target_chembl_id = target.get("target_chembl_id")
            target_name = target.get("pref_name")
            target_type = target.get("target_type")

            print(f"  Downloading activities for {target_chembl_id}: {target_name}", flush=True)
            try:
                activities = activity_client.filter(
                    target_chembl_id=target_chembl_id,
                    standard_type__in=standard_types,
                ).only(
                    [
                        "molecule_chembl_id",
                        "canonical_smiles",
                        "standard_type",
                        "standard_value",
                        "standard_units",
                        "pchembl_value",
                        "target_chembl_id",
                        "document_chembl_id",
                    ]
                )

                target_rows = []
                for activity in activities:
                    target_rows.append(
                        {
                            "query_gene": gene,
                            "query_uniprot_id": uniprot_id,
                            "target_chembl_id": target_chembl_id,
                            "target_name": target_name,
                            "target_type": target_type,
                            "organism": "Homo sapiens",
                            **activity,
                        }
                    )
            except Exception as exc:
                print(f"  Activity download failed for {target_chembl_id}: {exc}", flush=True)
                continue

            rows.extend(target_rows)
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"  Added {len(target_rows):,} rows; partial file saved", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"\nSaved {len(df):,} rows to {out_path}", flush=True)
    if not df.empty:
        print("\nTargets found:")
        print(
            df[["query_gene", "query_uniprot_id", "target_chembl_id", "target_name"]]
            .drop_duplicates()
            .to_string(index=False)
        )


def download_alphafold_for(session, uniprot_ids: list[str], force: bool = False) -> list[dict]:
    """Fetch AlphaFold PDBs for UniProt IDs not yet on disk and merge them into metadata.csv."""
    import csv

    from data.sources.alphafold import alphafold_download_structures
    from multimodal_datapipeline.utils.io import write_csv

    af_root = PROJECT_ROOT / "dataset_pipeline_output" / "alphafold"
    structure_dir = af_root / "structures"
    metadata_csv = af_root / "metadata.csv"
    existing = {path.stem for path in structure_dir.glob("*.pdb")} if structure_dir.exists() else set()
    to_download = sorted({uid for uid in uniprot_ids if uid and (force or uid not in existing)})
    print(f"AlphaFold: {len(to_download)} new structure(s) to download", flush=True)
    new_rows = alphafold_download_structures(session, to_download, str(structure_dir)) if to_download else []
    for row in new_rows:
        if row.get("local_pdb_path"):
            row["local_pdb_path"] = str(Path(row["local_pdb_path"]).relative_to(PROJECT_ROOT))

    rows_by_id = {}
    if metadata_csv.exists() and metadata_csv.stat().st_size:
        with metadata_csv.open(newline="") as handle:
            rows_by_id = {row["uniprot_id"]: row for row in csv.DictReader(handle)}
    for row in new_rows:
        rows_by_id[row["uniprot_id"]] = row
    write_csv(str(metadata_csv), list(rows_by_id.values()))
    failed = [row["uniprot_id"] for row in new_rows if row.get("status") != "downloaded"]
    if failed:
        print(f"AlphaFold: {len(failed)} ID(s) without a PDB: {failed}", flush=True)
    return new_rows


def run_davis(argv: list[str]) -> None:
    """Download the Davis benchmark, map kinases to UniProt and fetch AlphaFold structures."""
    import argparse
    import csv

    import requests

    from data.sources.davis import acquire_davis

    parser = argparse.ArgumentParser(prog="acquire_data.py davis")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-identity", type=float, default=0.9)
    parser.add_argument("--skip-alphafold", action="store_true")
    args = parser.parse_args(argv)

    outdir = PROJECT_ROOT / "dataset_pipeline_output" / "davis"
    with requests.Session() as session:
        pairs_csv, mapping_csv = acquire_davis(session, str(outdir), force=args.force, min_identity=args.min_identity)
        with open(mapping_csv, newline="") as handle:
            mapping = list(csv.DictReader(handle))
        mapped = [row for row in mapping if row["status"] == "mapped"]
        print(f"Davis targets mapped to UniProt: {len(mapped)}/{len(mapping)}", flush=True)
        if not args.skip_alphafold:
            download_alphafold_for(session, [row["uniprot_id"] for row in mapped], force=args.force)
    print(f"Wrote {pairs_csv}\nWrote {mapping_csv}")


def run_bbbc021_chembl(argv: list[str]) -> None:
    """Link BBBC021 compounds to ChEMBL binding activities, targets and AlphaFold structures."""
    import argparse
    import csv

    import requests
    from rdkit import Chem, RDLogger

    from data.sources.chembl import (
        chembl_fetch_molecule_activities,
        chembl_fetch_target,
        chembl_find_molecules_by_inchikey,
    )
    from multimodal_datapipeline.utils.io import write_csv

    parser = argparse.ArgumentParser(prog="acquire_data.py bbbc021-chembl")
    parser.add_argument("--max-activities-per-compound", type=int, default=5000)
    parser.add_argument("--skip-alphafold", action="store_true")
    args = parser.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")

    compound_csv = PROJECT_ROOT / "dataset_pipeline_output" / "bbbc021" / "BBBC021_v1_compound.csv"
    out_dir = PROJECT_ROOT / "dataset_pipeline_output" / "chembl"
    out_dir.mkdir(parents=True, exist_ok=True)
    with compound_csv.open(newline="") as handle:
        compounds = [row for row in csv.DictReader(handle) if row.get("smiles")]

    map_rows, activity_rows = [], []
    with requests.Session() as session:
        for compound in compounds:
            name, smiles = compound["compound"], compound["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                map_rows.append({"compound": name, "smiles": smiles, "status": "invalid_smiles"})
                continue
            inchikey = Chem.MolToInchiKey(mol)
            try:
                molecules = chembl_find_molecules_by_inchikey(session, inchikey)
            except Exception as exc:
                map_rows.append({"compound": name, "smiles": smiles, "inchikey": inchikey, "status": f"failed: {exc}"})
                continue
            if not molecules:
                map_rows.append({"compound": name, "smiles": smiles, "inchikey": inchikey, "status": "not_in_chembl"})
                continue

            chosen = molecules[0]
            activities = chembl_fetch_molecule_activities(
                session, chosen["molecule_chembl_id"], max_records=args.max_activities_per_compound
            )
            print(f"{name}: {chosen['molecule_chembl_id']} ({chosen['match']}) {len(activities)} activities", flush=True)
            map_rows.append(
                {
                    "compound": name,
                    "smiles": smiles,
                    "inchikey": inchikey,
                    "molecule_chembl_id": chosen["molecule_chembl_id"],
                    "chembl_pref_name": chosen["pref_name"],
                    "match": chosen["match"],
                    "n_activities": len(activities),
                    "status": "mapped",
                }
            )
            for activity in activities:
                activity_rows.append({"compound": name, "bbbc021_smiles": smiles, **activity})

        targets = {}
        for target_id in sorted({row["target_chembl_id"] for row in activity_rows if row["target_chembl_id"]}):
            try:
                targets[target_id] = chembl_fetch_target(session, target_id)
            except Exception as exc:
                print(f"  target lookup failed for {target_id}: {exc}", flush=True)
        for row in activity_rows:
            target = targets.get(row["target_chembl_id"], {})
            row["target_type"] = target.get("target_type", "")
            row["uniprot_ids"] = target.get("uniprot_ids", "")

        write_csv(str(out_dir / "bbbc021_compound_chembl_map.csv"), map_rows)
        write_csv(str(out_dir / "bbbc021_compound_activities.csv"), activity_rows)
        single_protein_ids = {
            row["uniprot_ids"]
            for row in activity_rows
            if row["target_type"] == "SINGLE PROTEIN" and row["uniprot_ids"] and ";" not in row["uniprot_ids"]
        }
        print(
            f"Mapped {sum(r['status'] == 'mapped' for r in map_rows)}/{len(map_rows)} compounds; "
            f"{len(activity_rows)} activities; {len(single_protein_ids)} single-protein targets",
            flush=True,
        )
        if not args.skip_alphafold:
            download_alphafold_for(session, sorted(single_protein_ids))


def main() -> None:
    subcommands = {
        "davis": run_davis,
        "bbbc021-chembl": run_bbbc021_chembl,
    }
    if len(sys.argv) > 1 and sys.argv[1] in subcommands:
        subcommands[sys.argv[1]](sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "cytoskeleton-chembl":
        sys.argv.pop(1)
        if len(sys.argv) > 1:
            raise SystemExit("cytoskeleton-chembl does not accept additional arguments")
        run_cytoskeleton_chembl()
        return

    run_dataset_pipeline()


if __name__ == "__main__":
    main()
