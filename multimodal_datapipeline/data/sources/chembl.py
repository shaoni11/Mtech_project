import time
from urllib.parse import urljoin


CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data/"


def chembl_fetch_page(session, endpoint, params):
    url = urljoin(CHEMBL_BASE, endpoint)
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def chembl_fetch_activities(session, target_chembl_id, standard_type=None, limit=1000, max_records=None):
    rows_out = []
    offset = 0
    fetched = 0

    while True:
        params = {
            "target_chembl_id": target_chembl_id,
            "limit": limit,
            "offset": offset,
            "format": "json",
        }
        if standard_type:
            params["standard_type"] = standard_type

        data = chembl_fetch_page(session, "activity.json", params)
        rows = data.get("activities", [])
        if not rows:
            break

        for row in rows:
            rows_out.append(
                {
                    "activity_id": row.get("activity_id"),
                    "assay_chembl_id": row.get("assay_chembl_id"),
                    "target_chembl_id": row.get("target_chembl_id"),
                    "molecule_chembl_id": row.get("molecule_chembl_id"),
                    "canonical_smiles": row.get("canonical_smiles"),
                    "standard_type": row.get("standard_type"),
                    "standard_relation": row.get("standard_relation"),
                    "standard_value": row.get("standard_value"),
                    "standard_units": row.get("standard_units"),
                    "pchembl_value": row.get("pchembl_value"),
                    "bao_label": row.get("bao_label"),
                    "document_chembl_id": row.get("document_chembl_id"),
                }
            )
            fetched += 1
            if max_records and fetched >= max_records:
                return rows_out

        if len(rows) < limit:
            break

        offset += limit
        time.sleep(0.2)

    return rows_out


def chembl_fetch_targets_for_uniprot(session, uniprot_id, limit=100):
    """Return ChEMBL targets linked to a UniProt accession."""
    data = chembl_fetch_page(
        session,
        "target.json",
        {
            "target_components__accession": uniprot_id,
            "limit": limit,
            "format": "json",
        },
    )
    rows = []
    for target in data.get("targets", []):
        rows.append(
            {
                "uniprot_id": uniprot_id,
                "target_chembl_id": target.get("target_chembl_id"),
                "pref_name": target.get("pref_name"),
                "target_type": target.get("target_type"),
                "organism": target.get("organism"),
            }
        )
    return rows
