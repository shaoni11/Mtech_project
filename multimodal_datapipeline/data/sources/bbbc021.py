import os
import zipfile

from multimodal_datapipeline.utils.io import ensure_dir


BBBC021_BASE = "https://data.broadinstitute.org/bbbc/BBBC021/"

BBBC021_METADATA_FILES = [
    "BBBC021_v1_image.csv",
    "BBBC021_v1_compound.csv",
    "BBBC021_v1_moa.csv",
]

BBBC021_WEEK1_ZIPS = [
    "BBBC021_v1_images_Week1_22123.zip",
    "BBBC021_v1_images_Week1_22141.zip",
    "BBBC021_v1_images_Week1_22161.zip",
    "BBBC021_v1_images_Week1_22361.zip",
    "BBBC021_v1_images_Week1_22381.zip",
    "BBBC021_v1_images_Week1_22401.zip",
]


def file_is_present(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_file_if_missing(session, url, out_path, force=False):
    if file_is_present(out_path) and not force:
        return "skipped_existing"

    ensure_dir(os.path.dirname(out_path))
    tmp_path = out_path + ".part"
    response = session.get(url, stream=True, timeout=180)
    response.raise_for_status()
    with open(tmp_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    os.replace(tmp_path, out_path)
    return "downloaded"


def extract_zip_if_missing(zip_path, extract_dir, force=False):
    marker_path = os.path.join(extract_dir, ".extract_complete")
    if file_is_present(marker_path) and not force:
        return "skipped_existing"

    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    with open(marker_path, "w", encoding="utf-8") as handle:
        handle.write(os.path.basename(zip_path) + "\n")
    return "extracted"


def bbbc021_download_metadata(session, outdir, force=False):
    rows = []
    ensure_dir(outdir)
    for filename in BBBC021_METADATA_FILES:
        out_path = os.path.join(outdir, filename)
        status = download_file_if_missing(session, BBBC021_BASE + filename, out_path, force=force)
        rows.append(
            {
                "dataset": "BBBC021",
                "kind": "metadata",
                "filename": filename,
                "status": status,
                "file": out_path,
            }
        )
    return rows


def bbbc021_download_images(session, outdir, image_zips=None, max_zips=None, extract=False, force=False):
    rows = []
    ensure_dir(outdir)
    selected = list(image_zips or BBBC021_WEEK1_ZIPS)
    if max_zips is not None:
        selected = selected[:max_zips]

    zip_dir = os.path.join(outdir, "zips")
    image_dir = os.path.join(outdir, "images")

    for filename in selected:
        zip_path = os.path.join(zip_dir, filename)
        status = download_file_if_missing(session, BBBC021_BASE + filename, zip_path, force=force)
        extract_status = "not_requested"
        if extract:
            extract_status = extract_zip_if_missing(
                zip_path,
                os.path.join(image_dir, filename.removesuffix(".zip")),
                force=force,
            )
        rows.append(
            {
                "dataset": "BBBC021",
                "kind": "image_zip",
                "filename": filename,
                "status": status,
                "extract_status": extract_status,
                "file": zip_path,
            }
        )
    return rows
