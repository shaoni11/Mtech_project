"""Davis kinase-inhibitor benchmark (68 drugs x 442 kinases, Kd in nM).

Downloads the plain-text DeepPurpose release of Davis (no pickles), maps each
kinase construct to a reviewed human UniProt accession so AlphaFold structures
can be fetched, and writes a long-format pair table.
"""

import io
import json
import math
import os
import re
import time
import zipfile
from difflib import SequenceMatcher

from multimodal_datapipeline.utils.io import ensure_dir, write_csv


DAVIS_ZIP_URL = "https://github.com/futianfan/DeepPurpose_Data/blob/main/DAVIS.zip?raw=true"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Davis construct names that are not HGNC gene symbols.
DAVIS_NAME_ALIASES = {
    "AMPK-alpha1": "PRKAA1",
    "AMPK-alpha2": "PRKAA2",
    "IKK-alpha": "CHUK",
    "IKK-beta": "IKBKB",
    "IKK-epsilon": "IKBKE",
    "PKAC-alpha": "PRKACA",
    "PKAC-beta": "PRKACB",
    "ERK1": "MAPK3",
    "ERK2": "MAPK1",
    "ERK3": "MAPK6",
    "ERK4": "MAPK4",
    "ERK5": "MAPK7",
    "ERK8": "MAPK15",
    "JNK1": "MAPK8",
    "JNK2": "MAPK9",
    "JNK3": "MAPK10",
    "p38-alpha": "MAPK14",
    "p38-beta": "MAPK11",
    "p38-gamma": "MAPK12",
    "p38-delta": "MAPK13",
    "GCN2": "EIF2AK4",
    "PKR": "EIF2AK2",
    "PERK": "EIF2AK3",
    "HRI": "EIF2AK1",
    "LOK": "STK10",
    "YSK1": "STK25",
    "YSK4": "MAP3K19",
    "MST1": "STK4",
    "MST2": "STK3",
    "MST3": "STK24",
    "MST4": "STK26",
    "TAK1": "MAP3K7",
    "MLK1": "MAP3K9",
    "MLK2": "MAP3K10",
    "MLK3": "MAP3K11",
    "MEK1": "MAP2K1",
    "MEK2": "MAP2K2",
    "MEK3": "MAP2K3",
    "MEK4": "MAP2K4",
    "MEK5": "MAP2K5",
    "MEK6": "MAP2K6",
    "MKK7": "MAP2K7",
    "HPK1": "MAP4K1",
    "GCK": "MAP4K2",
    "HGK": "MAP4K4",
    "KHS1": "MAP4K5",
    "MINK": "MINK1",
    "CK1-alpha1": "CSNK1A1",
    "CK1-alpha1L": "CSNK1A1L",
    "CK1-delta": "CSNK1D",
    "CK1-epsilon": "CSNK1E",
    "CK1-gamma1": "CSNK1G1",
    "CK1-gamma2": "CSNK1G2",
    "CK1-gamma3": "CSNK1G3",
    "CK2-alpha": "CSNK2A1",
    "CK2-alpha2": "CSNK2A2",
    "DCAMKL1": "DCLK1",
    "DCAMKL2": "DCLK2",
    "DCAMKL3": "DCLK3",
    "PFTAIRE2": "CDK15",
    "PFTK1": "CDK14",
    "PCTK1": "CDK16",
    "PCTK2": "CDK17",
    "PCTK3": "CDK18",
    "CDC2L1": "CDK11B",
    "CDC2L2": "CDK11A",
    "CDC2L5": "CDK13",
    "CHK1": "CHEK1",
    "CHK2": "CHEK2",
    "PIP5K2B": "PIP4K2B",
    "PIP5K2C": "PIP4K2C",
    "PIK4CB": "PI4KB",
    "RSK1": "RPS6KA1",
    "RSK2": "RPS6KA3",
    "RSK3": "RPS6KA2",
    "RSK4": "RPS6KA6",
    "MSK1": "RPS6KA5",
    "MSK2": "RPS6KA4",
    "S6K1": "RPS6KB1",
    "SgK085": "SGK085",
    "SgK110": "SGK110",
    "TRKA": "NTRK1",
    "TRKB": "NTRK2",
    "TRKC": "NTRK3",
    "SNARK": "NUAK2",
    "MRCKA": "CDC42BPA",
    "MRCKB": "CDC42BPB",
    "ARK5": "NUAK1",
    "BIKE": "BMP2K",
    "DLK": "MAP3K12",
    "LZK": "MAP3K13",
    "ZAK": "MAP3K20",
    "MEKK1": "MAP3K1",
    "MEKK2": "MAP3K2",
    "MEKK3": "MAP3K3",
    "MEKK6": "MAP3K6",
    "ASK1": "MAP3K5",
    "ASK2": "MAP3K6",
    "OSR1": "OXSR1",
    "PAK7": "PAK5",
    "PRKR": "EIF2AK2",
    "RIPK5": "DSTYK",
    "YANK1": "STK32A",
    "YANK2": "STK32B",
    "YANK3": "STK32C",
    "PRP4": "PRPF4B",
    "LKB1": "STK11",
    "ADCK3": "COQ8A",
    "ADCK4": "COQ8B",
}


def davis_base_name(name):
    """'ABL1(T315I)p' -> 'ABL1'; 'CDK4-cyclinD1' -> 'CDK4-cyclinD1' (resolved by candidates)."""
    base = re.sub(r"\(.*?\)", "", name)
    if base.endswith("p") and base[:-1] and base[:-1][-1].isalnum() and base[:-1].upper() == base[:-1]:
        base = base[:-1]  # trailing 'p' marks a phosphorylated construct
    return base


def davis_name_candidates(name):
    base = davis_base_name(name)
    candidates = []
    for value in [
        DAVIS_NAME_ALIASES.get(name),
        DAVIS_NAME_ALIASES.get(base),
        base,
        base.split("-")[0],
        DAVIS_NAME_ALIASES.get(base.split("-")[0]),
        re.sub(r"[^A-Za-z0-9]", "", base).upper(),
    ]:
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def download_davis(session, outdir, force=False):
    """Download and parse Davis into (drug_id, smiles), (target, sequence) and the Kd matrix."""
    ensure_dir(outdir)
    zip_path = os.path.join(outdir, "DAVIS.zip")
    if force or not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        response = session.get(DAVIS_ZIP_URL, timeout=120)
        response.raise_for_status()
        with open(zip_path, "wb") as handle:
            handle.write(response.content)

    with zipfile.ZipFile(zip_path) as archive:
        def read_member(suffix):
            member = next(n for n in archive.namelist() if n.endswith(suffix) and "__MACOSX" not in n)
            return archive.read(member).decode("utf-8")

        drugs = json.loads(read_member("SMILES.txt"))
        targets = json.loads(read_member("target_seq.txt"))
        affinity = [
            [float(value) for value in line.split()]
            for line in io.StringIO(read_member("affinity.txt"))
            if line.strip()
        ]

    drug_items = list(drugs.items())
    target_items = list(targets.items())
    if len(affinity) != len(drug_items) or any(len(row) != len(target_items) for row in affinity):
        raise ValueError("Davis affinity matrix shape does not match drug/target lists.")
    return drug_items, target_items, affinity


def sequence_identity(query, reference):
    if len(query) == len(reference):
        return sum(a == b for a, b in zip(query, reference)) / max(1, len(query))
    matcher = SequenceMatcher(None, query, reference, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / max(1, len(query))


def uniprot_candidates(session, gene):
    params = {
        "query": f"(gene_exact:{gene}) AND organism_id:9606 AND reviewed:true",
        "fields": "accession,gene_primary,sequence",
        "format": "tsv",
        "size": 10,
    }
    response = session.get(UNIPROT_SEARCH, params=params, timeout=60)
    response.raise_for_status()
    lines = response.text.strip().splitlines()[1:]
    return [line.split("\t") for line in lines if line.count("\t") >= 2]


def map_davis_targets_to_uniprot(session, target_items, min_identity=0.9):
    """Map each Davis construct to the reviewed human UniProt entry it matches best by sequence."""
    rows = []
    cache = {}
    for name, sequence in target_items:
        best = None
        tried = davis_name_candidates(name)
        for gene in tried:
            if gene not in cache:
                try:
                    cache[gene] = uniprot_candidates(session, gene)
                except Exception as exc:
                    print(f"  UniProt lookup failed for {gene}: {exc}", flush=True)
                    cache[gene] = []
                time.sleep(0.05)
            for accession, gene_primary, reference in cache[gene]:
                identity = sequence_identity(sequence, reference)
                if best is None or identity > best[2]:
                    best = (accession, gene_primary, identity, len(reference))
            if best and best[2] >= min_identity:
                break

        mapped = best is not None and best[2] >= min_identity
        rows.append(
            {
                "davis_target": name,
                "query_names": ";".join(tried),
                "uniprot_id": best[0] if mapped else "",
                "gene": best[1] if mapped else "",
                "sequence_identity": f"{best[2]:.4f}" if best else "",
                "davis_sequence_length": len(sequence),
                "uniprot_sequence_length": best[3] if best else "",
                "status": "mapped" if mapped else "unmapped",
            }
        )
    return rows


def davis_pair_rows(drug_items, target_items, affinity):
    rows = []
    for drug_index, (drug_id, smiles) in enumerate(drug_items):
        for target_index, (target_name, sequence) in enumerate(target_items):
            kd_nm = affinity[drug_index][target_index]
            rows.append(
                {
                    "drug_id": drug_id,
                    "smiles": smiles,
                    "davis_target": target_name,
                    "protein_sequence": sequence,
                    "kd_nm": kd_nm,
                    "pkd": f"{-math.log10(kd_nm / 1e9):.4f}",
                }
            )
    return rows


def acquire_davis(session, outdir, force=False, min_identity=0.9):
    drug_items, target_items, affinity = download_davis(session, outdir, force=force)
    print(f"Davis: {len(drug_items)} drugs x {len(target_items)} targets", flush=True)

    pairs_csv = os.path.join(outdir, "davis_pairs.csv")
    write_csv(pairs_csv, davis_pair_rows(drug_items, target_items, affinity))

    mapping_csv = os.path.join(outdir, "davis_target_uniprot.csv")
    if os.path.exists(mapping_csv) and not force:
        print("Found existing Davis UniProt mapping; skipping lookup:", mapping_csv, flush=True)
    else:
        print("Mapping Davis targets to UniProt", flush=True)
        write_csv(mapping_csv, map_davis_targets_to_uniprot(session, target_items, min_identity=min_identity))
    return pairs_csv, mapping_csv
