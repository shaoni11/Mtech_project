import os
import time
import zipfile

from multimodal_datapipeline.utils.io import ensure_dir


BBBC021_BASE = "https://data.broadinstitute.org/bbbc/BBBC021/"
BBBC021_HF_MIRROR_BASE = "https://huggingface.co/datasets/roslu/BBBC021-Human-MCF7-Cells/resolve/main/"

BBBC021_METADATA_FILES = [
    "BBBC021_v1_image.csv",
    "BBBC021_v1_compound.csv",
    "BBBC021_v1_moa.csv",
]

BBBC021_CELLPROFILER_FILES = [
    "analysis.cppipe",
    "illum.cppipe",
]

BBBC021_WEEK1_ZIPS = [
    "BBBC021_v1_images_Week1_22123.zip",
    "BBBC021_v1_images_Week1_22141.zip",
    "BBBC021_v1_images_Week1_22161.zip",
    "BBBC021_v1_images_Week1_22361.zip",
    "BBBC021_v1_images_Week1_22381.zip",
    "BBBC021_v1_images_Week1_22401.zip",
]

BBBC021_WEEK_ZIPS = {
    "Week1": BBBC021_WEEK1_ZIPS,
    "Week2": [
        "BBBC021_v1_images_Week2_24121.zip",
        "BBBC021_v1_images_Week2_24141.zip",
        "BBBC021_v1_images_Week2_24161.zip",
        "BBBC021_v1_images_Week2_24361.zip",
        "BBBC021_v1_images_Week2_24381.zip",
        "BBBC021_v1_images_Week2_24401.zip",
    ],
    "Week3": [
        "BBBC021_v1_images_Week3_25421.zip",
        "BBBC021_v1_images_Week3_25441.zip",
        "BBBC021_v1_images_Week3_25461.zip",
        "BBBC021_v1_images_Week3_25681.zip",
        "BBBC021_v1_images_Week3_25701.zip",
        "BBBC021_v1_images_Week3_25721.zip",
    ],
    "Week4": [
        "BBBC021_v1_images_Week4_27481.zip",
        "BBBC021_v1_images_Week4_27521.zip",
        "BBBC021_v1_images_Week4_27542.zip",
        "BBBC021_v1_images_Week4_27801.zip",
        "BBBC021_v1_images_Week4_27821.zip",
        "BBBC021_v1_images_Week4_27861.zip",
    ],
    "Week5": [
        "BBBC021_v1_images_Week5_28901.zip",
        "BBBC021_v1_images_Week5_28921.zip",
        "BBBC021_v1_images_Week5_28961.zip",
        "BBBC021_v1_images_Week5_29301.zip",
        "BBBC021_v1_images_Week5_29321.zip",
        "BBBC021_v1_images_Week5_29341.zip",
    ],
    "Week6": [
        "BBBC021_v1_images_Week6_31641.zip",
        "BBBC021_v1_images_Week6_31661.zip",
        "BBBC021_v1_images_Week6_31681.zip",
        "BBBC021_v1_images_Week6_32061.zip",
        "BBBC021_v1_images_Week6_32121.zip",
        "BBBC021_v1_images_Week6_32161.zip",
    ],
    "Week7": [
        "BBBC021_v1_images_Week7_34341.zip",
        "BBBC021_v1_images_Week7_34381.zip",
        "BBBC021_v1_images_Week7_34641.zip",
        "BBBC021_v1_images_Week7_34661.zip",
        "BBBC021_v1_images_Week7_34681.zip",
    ],
    "Week8": [
        "BBBC021_v1_images_Week8_38203.zip",
        "BBBC021_v1_images_Week8_38221.zip",
        "BBBC021_v1_images_Week8_38241.zip",
        "BBBC021_v1_images_Week8_38341.zip",
        "BBBC021_v1_images_Week8_38342.zip",
    ],
    "Week9": [
        "BBBC021_v1_images_Week9_39206.zip",
        "BBBC021_v1_images_Week9_39221.zip",
        "BBBC021_v1_images_Week9_39222.zip",
        "BBBC021_v1_images_Week9_39282.zip",
        "BBBC021_v1_images_Week9_39283.zip",
        "BBBC021_v1_images_Week9_39301.zip",
    ],
    "Week10": [
        "BBBC021_v1_images_Week10_40111.zip",
        "BBBC021_v1_images_Week10_40115.zip",
        "BBBC021_v1_images_Week10_40119.zip",
    ],
}

BBBC021_ALL_IMAGE_ZIPS = [
    filename
    for week_zips in BBBC021_WEEK_ZIPS.values()
    for filename in week_zips
]


def file_is_present(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_file_if_missing(session, url, out_path, force=False, retries=5):
    if file_is_present(out_path) and not force:
        return "skipped_existing"

    ensure_dir(os.path.dirname(out_path))
    tmp_path = out_path + ".part"
    if force and os.path.exists(tmp_path):
        os.remove(tmp_path)
    headers = {"User-Agent": "multimodal-datapipeline/1.0"}
    last_exc = None
    urls = url if isinstance(url, list) else [url]
    for source_index, source_url in enumerate(urls, start=1):
        for attempt in range(1, retries + 1):
            try:
                request_headers = dict(headers)
                existing_bytes = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
                if existing_bytes:
                    request_headers["Range"] = f"bytes={existing_bytes}-"
                    print(
                        f"  resuming {os.path.basename(out_path)} from {existing_bytes / (1024 * 1024):.1f} MB",
                        flush=True,
                    )

                response = session.get(source_url, stream=True, timeout=180, headers=request_headers)
                response.raise_for_status()
                mode = "ab" if existing_bytes and response.status_code == 206 else "wb"
                if existing_bytes and response.status_code != 206:
                    print(
                        f"  source did not honor resume for {os.path.basename(out_path)}; restarting file",
                        flush=True,
                    )
                with open(tmp_path, mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                os.replace(tmp_path, out_path)
                return "downloaded"
            except Exception as exc:
                last_exc = exc
                if attempt == retries:
                    break
                sleep_seconds = min(120, 10 * attempt)
                print(
                    f"Download failed for {os.path.basename(out_path)} "
                    f"(source {source_index}/{len(urls)}, attempt {attempt}/{retries}): "
                    f"{exc}. Retrying in {sleep_seconds}s.",
                    flush=True,
                )
                time.sleep(sleep_seconds)
        if source_index < len(urls):
            print(
                f"Switching download source for {os.path.basename(out_path)} "
                f"after error: {last_exc}",
                flush=True,
            )
    else:
        raise RuntimeError(f"Download failed for {urls}") from last_exc


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


def bbbc021_download_cellprofiler_pipelines(session, outdir, force=False):
    rows = []
    ensure_dir(outdir)
    for filename in BBBC021_CELLPROFILER_FILES:
        out_path = os.path.join(outdir, filename)
        status = download_file_if_missing(session, BBBC021_BASE + filename, out_path, force=force)
        rows.append(
            {
                "dataset": "BBBC021",
                "kind": "cellprofiler_pipeline",
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
        print(f"BBBC021 image ZIP: {filename}", flush=True)
        status = download_file_if_missing(
            session,
            [
                BBBC021_HF_MIRROR_BASE + filename + "?download=true",
                BBBC021_BASE + filename,
            ],
            zip_path,
            force=force,
        )
        extract_status = "not_requested"
        if extract:
            extract_status = extract_zip_if_missing(
                zip_path,
                os.path.join(image_dir, filename.removesuffix(".zip")),
                force=force,
            )
        print(f"  download={status} extract={extract_status}", flush=True)
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
