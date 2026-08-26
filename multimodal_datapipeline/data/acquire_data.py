#!/usr/bin/env python3
"""Unified data acquisition entrypoint.

Default mode delegates to the existing Phase 1 dataset pipeline without
changing its CLI arguments. The `cytoskeleton-chembl` subcommand runs the
actin/tubulin ChEMBL acquisition workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from chembl_webresource_client.new_client import new_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = PROJECT_ROOT / "package"
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


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "cytoskeleton-chembl":
        sys.argv.pop(1)
        if len(sys.argv) > 1:
            raise SystemExit("cytoskeleton-chembl does not accept additional arguments")
        run_cytoskeleton_chembl()
        return

    run_dataset_pipeline()


if __name__ == "__main__":
    main()
