import os
import time


ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/"


def alphafold_get_prediction(session, uniprot_id):
    response = session.get(ALPHAFOLD_API + uniprot_id, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not data:
        return None
    return data[0]


def download_file(session, url, out_path):
    response = session.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(out_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def alphafold_download_structures(session, ids, outdir):
    from multimodal_datapipeline.utils.io import ensure_dir

    ensure_dir(outdir)
    rows = []

    for uid in ids:
        try:
            meta = alphafold_get_prediction(session, uid)
            if not meta:
                rows.append({"uniprot_id": uid, "status": "missing"})
                continue

            pdb_url = meta.get("pdbUrl")
            cif_url = meta.get("cifUrl")
            out_pdb = os.path.join(outdir, f"{uid}.pdb") if pdb_url else ""

            if pdb_url:
                download_file(session, pdb_url, out_pdb)

            rows.append(
                {
                    "uniprot_id": uid,
                    "status": "downloaded" if pdb_url else "metadata_only",
                    "entryId": meta.get("entryId"),
                    "uniprotAccession": meta.get("uniprotAccession"),
                    "uniprotId": meta.get("uniprotId"),
                    "modelCreatedDate": meta.get("modelCreatedDate"),
                    "pdbUrl": pdb_url,
                    "cifUrl": cif_url,
                    "paeImageUrl": meta.get("paeImageUrl"),
                    "sequenceLength": meta.get("sequenceLength"),
                    "gene": meta.get("gene"),
                    "organismScientificName": meta.get("organismScientificName"),
                    "local_pdb_path": out_pdb,
                }
            )
            time.sleep(0.1)
        except Exception as exc:
            rows.append({"uniprot_id": uid, "status": f"failed: {exc}"})

    return rows
