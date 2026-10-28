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


def chembl_find_molecules_by_inchikey(session, inchikey):
    """Exact InChIKey match, falling back to the stereo-agnostic connectivity block."""
    for params in (
        {"molecule_structures__standard_inchi_key": inchikey},
        {"molecule_structures__standard_inchi_key__startswith": inchikey.split("-")[0]},
    ):
        data = chembl_fetch_page(session, "molecule.json", {**params, "limit": 20, "format": "json"})
        molecules = data.get("molecules", [])
        if molecules:
            match = "exact" if "molecule_structures__standard_inchi_key" in params else "connectivity"
            return [
                {
                    "molecule_chembl_id": molecule.get("molecule_chembl_id"),
                    "pref_name": molecule.get("pref_name"),
                    "match": match,
                }
                for molecule in molecules
            ]
    return []


def chembl_fetch_molecule_activities(session, molecule_chembl_id, limit=1000, max_records=None):
    """Human binding-assay activities with a pChEMBL value for one molecule."""
    rows_out = []
    offset = 0
    while True:
        data = chembl_fetch_page(
            session,
            "activity.json",
            {
                "molecule_chembl_id": molecule_chembl_id,
                "assay_type": "B",
                "target_organism": "Homo sapiens",
                "pchembl_value__isnull": "false",
                "limit": limit,
                "offset": offset,
                "format": "json",
            },
        )
        rows = data.get("activities", [])
        for row in rows:
            rows_out.append(
                {
                    "activity_id": row.get("activity_id"),
                    "molecule_chembl_id": row.get("molecule_chembl_id"),
                    "target_chembl_id": row.get("target_chembl_id"),
                    "target_pref_name": row.get("target_pref_name"),
                    "standard_type": row.get("standard_type"),
                    "standard_relation": row.get("standard_relation"),
                    "standard_value": row.get("standard_value"),
                    "standard_units": row.get("standard_units"),
                    "pchembl_value": row.get("pchembl_value"),
                    "assay_chembl_id": row.get("assay_chembl_id"),
                    "document_chembl_id": row.get("document_chembl_id"),
                }
            )
            if max_records and len(rows_out) >= max_records:
                return rows_out
        if len(rows) < limit:
            return rows_out
        offset += limit
        time.sleep(0.2)


def chembl_fetch_target(session, target_chembl_id):
    data = chembl_fetch_page(session, f"target/{target_chembl_id}.json", {"format": "json"})
    accessions = [
        component.get("accession")
        for component in data.get("target_components", [])
        if component.get("accession")
    ]
    return {
        "target_chembl_id": target_chembl_id,
        "pref_name": data.get("pref_name"),
        "target_type": data.get("target_type"),
        "organism": data.get("organism"),
        "uniprot_ids": ";".join(accessions),
    }
