import argparse
import csv
import json
import os

import requests

from multimodal_datapipeline.data.alphafold import alphafold_download_structures
from multimodal_datapipeline.data.bbbc021 import (
    BBBC021_WEEK1_ZIPS,
    bbbc021_download_images,
    bbbc021_download_metadata,
)
from multimodal_datapipeline.data.chembl import chembl_fetch_activities, chembl_fetch_targets_for_uniprot
from multimodal_datapipeline.data.scrape import scrape_html_table
from multimodal_datapipeline.utils.io import ensure_dir, save_json, write_csv, write_table_csv


DEFAULT_PHASE1_UNIPROT_IDS = [
    "P00533",  # EGFR
    "P31749",  # AKT1
    "P31751",  # AKT2
    "Q9Y243",  # AKT3
    "P15056",  # BRAF
    "P28482",  # MAPK1/ERK2
    "P27361",  # MAPK3/ERK1
    "P24941",  # CDK2
    "P35968",  # VEGFR2/KDR
    "P12931",  # SRC
    "P00519",  # ABL1
    "P42345",  # MTOR
]


def file_is_present(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_csv_rows(path):
    if not file_is_present(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing_csv_targets(path):
    rows = read_csv_rows(path)
    return {row.get("target_chembl_id") for row in rows if row.get("target_chembl_id")}


def write_empty_file(path):
    with open(path, "w", encoding="utf-8"):
        pass


def merge_csv_files(out_path, input_paths):
    merged = []
    fieldnames = None
    for input_path in input_paths:
        rows = read_csv_rows(input_path)
        if not rows:
            continue
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        merged.extend(rows)
    if merged:
        write_csv(out_path, merged, fieldnames=fieldnames)
    else:
        write_empty_file(out_path)
    return len(merged)


def choose_primary_targets(mapping_rows):
    primary_by_uniprot = {}
    for row in mapping_rows:
        if row.get("target_type") != "SINGLE PROTEIN":
            continue
        primary_by_uniprot.setdefault(row["uniprot_id"], row)
    return list(primary_by_uniprot.values())


def main():
    parser = argparse.ArgumentParser(
        description="Master dataset pipeline: ChEMBL API + AlphaFold API + optional scraping"
    )
    parser.add_argument("--outdir", default="dataset_pipeline_output", help="Base output directory")
    parser.add_argument(
        "--download-missing-phase1",
        action="store_true",
        help="Download Phase 1 missing data: multi-target ChEMBL, target mapping, AlphaFold, and BBBC021 metadata.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when matching output files already exist.",
    )
    parser.add_argument("--chembl-target", default=None, help="ChEMBL target ID, e.g. CHEMBL203")
    parser.add_argument("--chembl-standard-type", default=None, help="IC50, Ki, Kd, etc.")
    parser.add_argument("--chembl-max-records", type=int, default=None)
    parser.add_argument(
        "--phase1-uniprot-ids",
        nargs="*",
        default=DEFAULT_PHASE1_UNIPROT_IDS,
        help="UniProt IDs used for Phase 1 ChEMBL target lookup and AlphaFold download.",
    )
    parser.add_argument("--alphafold-ids", nargs="*", default=[], help="UniProt IDs for AlphaFold download")
    parser.add_argument(
        "--download-bbbc021-images",
        action="store_true",
        help="Download BBBC021 image ZIPs. This is large; metadata is downloaded without this flag.",
    )
    parser.add_argument(
        "--bbbc021-image-zips",
        nargs="*",
        default=BBBC021_WEEK1_ZIPS,
        help="Specific BBBC021 image ZIP filenames to download.",
    )
    parser.add_argument(
        "--bbbc021-max-zips",
        type=int,
        default=None,
        help="Limit how many BBBC021 image ZIP files are downloaded.",
    )
    parser.add_argument(
        "--bbbc021-extract",
        action="store_true",
        help="Extract downloaded BBBC021 image ZIPs after download.",
    )
    parser.add_argument("--scrape-url", default=None, help="Optional HTML page with table to scrape")
    parser.add_argument("--scrape-table-index", type=int, default=0)
    args = parser.parse_args()

    outdir_abs = os.path.abspath(args.outdir)
    print("Output directory:", outdir_abs)
    print("Running with args:", args)

    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "chembl"))
    ensure_dir(os.path.join(args.outdir, "alphafold"))
    ensure_dir(os.path.join(args.outdir, "bbbc021"))
    ensure_dir(os.path.join(args.outdir, "scrape"))

    manifest = {"chembl": None, "target_mapping": None, "alphafold": None, "bbbc021": None, "scrape": None}

    with requests.Session() as session:
        if args.download_missing_phase1:
            print("Downloading Phase 1 missing datasets")
            print("Looking up ChEMBL targets for UniProt IDs:", args.phase1_uniprot_ids)
            mapping_rows = []
            target_mapping_csv = os.path.join(args.outdir, "chembl", "target_mapping.csv")
            if file_is_present(target_mapping_csv) and not args.force:
                print("Found existing target mapping; skipping lookup:", target_mapping_csv)
                mapping_rows = read_csv_rows(target_mapping_csv)
            else:
                for uniprot_id in args.phase1_uniprot_ids:
                    mapping_rows.extend(chembl_fetch_targets_for_uniprot(session, uniprot_id))
                if mapping_rows:
                    write_csv(target_mapping_csv, mapping_rows)
                else:
                    write_empty_file(target_mapping_csv)
                print("Wrote target mapping CSV:", target_mapping_csv)

            primary_targets = choose_primary_targets(mapping_rows)
            target_ids = [row["target_chembl_id"] for row in primary_targets if row.get("target_chembl_id")]
            manifest["target_mapping"] = {
                "uniprot_ids": args.phase1_uniprot_ids,
                "rows": len(mapping_rows),
                "primary_single_protein_targets": len(primary_targets),
                "file": target_mapping_csv,
            }

            per_target_dir = os.path.join(args.outdir, "chembl", "activities_by_target")
            ensure_dir(per_target_dir)
            activity_files = []
            for target_id in target_ids:
                target_csv = os.path.join(per_target_dir, f"{target_id}.csv")
                if file_is_present(target_csv) and not args.force:
                    print("Found existing ChEMBL target activities; skipping:", target_csv)
                else:
                    print("Fetching ChEMBL activities for", target_id)
                    rows = chembl_fetch_activities(
                        session,
                        target_chembl_id=target_id,
                        standard_type=args.chembl_standard_type,
                        max_records=args.chembl_max_records,
                    )
                    if rows:
                        write_csv(target_csv, rows)
                    else:
                        write_empty_file(target_csv)
                    print("Wrote ChEMBL target CSV:", target_csv)
                activity_files.append(target_csv)

            combined_csv = os.path.join(args.outdir, "chembl", "activities_multitarget.csv")
            combined_rows = merge_csv_files(combined_csv, activity_files)
            print("Wrote combined multi-target ChEMBL CSV:", combined_csv)
            manifest["chembl"] = {
                "mode": "phase1_multitarget",
                "standard_type": args.chembl_standard_type,
                "max_records_per_target": args.chembl_max_records,
                "targets": target_ids,
                "target_count": len(target_ids),
                "rows": combined_rows,
                "file": combined_csv,
                "per_target_dir": per_target_dir,
            }

        if args.chembl_target:
            chembl_csv = os.path.join(args.outdir, "chembl", "activities.csv")
            if file_is_present(chembl_csv) and args.chembl_target in existing_csv_targets(chembl_csv) and not args.force:
                print("Found existing ChEMBL activities for", args.chembl_target, "; skipping:", chembl_csv)
                chembl_rows = read_csv_rows(chembl_csv)
            else:
                print("Fetching ChEMBL activities for", args.chembl_target)
                chembl_rows = chembl_fetch_activities(
                    session,
                    target_chembl_id=args.chembl_target,
                    standard_type=args.chembl_standard_type,
                    max_records=args.chembl_max_records,
                )
                if chembl_rows:
                    write_csv(chembl_csv, chembl_rows)
                else:
                    write_empty_file(chembl_csv)
                print("Wrote ChEMBL CSV:", chembl_csv)

            manifest["chembl_single_target"] = {
                "target": args.chembl_target,
                "standard_type": args.chembl_standard_type,
                "rows": len(chembl_rows),
                "file": chembl_csv,
            }
        else:
            print("No --chembl-target provided; skipping ChEMBL")

        alphafold_ids = list(dict.fromkeys(args.alphafold_ids))
        if args.download_missing_phase1:
            alphafold_ids = list(dict.fromkeys(alphafold_ids + args.phase1_uniprot_ids))

        if alphafold_ids:
            af_dir = os.path.join(args.outdir, "alphafold", "structures")
            af_csv = os.path.join(args.outdir, "alphafold", "metadata.csv")
            existing_ids = {
                os.path.splitext(filename)[0]
                for filename in os.listdir(af_dir)
                if filename.endswith(".pdb")
            } if os.path.isdir(af_dir) else set()
            ids_to_download = [uid for uid in alphafold_ids if args.force or uid not in existing_ids]
            if ids_to_download:
                print("Downloading AlphaFold for IDs:", ids_to_download)
                new_af_rows = alphafold_download_structures(session, ids_to_download, af_dir)
            else:
                print("All requested AlphaFold structures already exist; skipping downloads")
                new_af_rows = []

            existing_af_rows = [] if args.force else read_csv_rows(af_csv)
            existing_af_by_id = {row.get("uniprot_id"): row for row in existing_af_rows if row.get("uniprot_id")}
            for row in new_af_rows:
                existing_af_by_id[row.get("uniprot_id")] = row
            af_rows = list(existing_af_by_id.values())
            if af_rows:
                write_csv(af_csv, af_rows)
            else:
                write_empty_file(af_csv)
            print("Wrote AlphaFold CSV:", af_csv)

            manifest["alphafold"] = {
                "ids": alphafold_ids,
                "rows": len(af_rows),
                "file": af_csv,
                "structure_dir": af_dir,
            }
        else:
            print("No --alphafold-ids provided; skipping AlphaFold")

        if args.download_missing_phase1:
            bbbc_dir = os.path.join(args.outdir, "bbbc021")
            print("Downloading BBBC021 metadata")
            bbbc_rows = bbbc021_download_metadata(session, bbbc_dir, force=args.force)
            if args.download_bbbc021_images:
                print("Downloading BBBC021 image ZIPs")
                bbbc_rows.extend(
                    bbbc021_download_images(
                        session,
                        bbbc_dir,
                        image_zips=args.bbbc021_image_zips,
                        max_zips=args.bbbc021_max_zips,
                        extract=args.bbbc021_extract,
                        force=args.force,
                    )
                )
            else:
                print("Skipping BBBC021 image ZIPs; pass --download-bbbc021-images to fetch them")
            bbbc_manifest_csv = os.path.join(bbbc_dir, "download_manifest.csv")
            if bbbc_rows:
                write_csv(bbbc_manifest_csv, bbbc_rows)
            else:
                write_empty_file(bbbc_manifest_csv)
            manifest["bbbc021"] = {
                "metadata_downloaded": True,
                "images_requested": args.download_bbbc021_images,
                "image_zips": args.bbbc021_image_zips[: args.bbbc021_max_zips],
                "extract": args.bbbc021_extract,
                "rows": len(bbbc_rows),
                "file": bbbc_manifest_csv,
            }
        else:
            print("No --download-missing-phase1 provided; skipping BBBC021")

    if args.scrape_url:
        print("Scraping URL:", args.scrape_url)
        rows = scrape_html_table(args.scrape_url, args.scrape_table_index)
        scrape_csv = os.path.join(args.outdir, "scrape", "scraped_table.csv")
        if rows:
            write_table_csv(scrape_csv, rows)
        else:
            open(scrape_csv, "w", encoding="utf-8").close()
        print("Wrote scraped table CSV:", scrape_csv)

        manifest["scrape"] = {
            "url": args.scrape_url,
            "table_index": args.scrape_table_index,
            "rows": len(rows),
            "file": scrape_csv,
        }
    else:
        print("No --scrape-url provided; skipping scraping")

    manifest_path = os.path.join(args.outdir, "manifest.json")
    save_json(manifest_path, manifest)
    print("Wrote manifest:", manifest_path)
    print(json.dumps(manifest, indent=2))
