"""BBBC021 microscopy loading and phenotype-profile helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


CHANNEL_COLUMNS = ["dapi_path", "tubulin_path", "actin_path"]


def load_channel(path: str, image_size: int) -> np.ndarray:
    """Resize one 16-bit channel and rescale its 1st-99th percentile range to [0, 1]."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32)

    if array.ndim == 3:
        array = array[..., 0]

    finite = np.isfinite(array)
    if not finite.all():
        array = np.where(finite, array, 0.0)

    low, high = np.percentile(array, [1.0, 99.0])
    if high <= low:
        high = float(array.max())
        low = float(array.min())
    if high > low:
        array = np.clip(array, low, high)
        array = (array - low) / (high - low)
    else:
        array = np.zeros_like(array, dtype=np.float32)

    return array.astype(np.float32)


def load_three_channel_array(paths: Sequence[str], image_size: int) -> np.ndarray:
    """DAPI, tubulin, actin -> float32 array [3, image_size, image_size]."""
    return np.stack([load_channel(path, image_size) for path in paths], axis=0)


def read_embedding_table(npz_path: Path, metadata_csv: Path) -> tuple[np.ndarray, list[dict[str, str]]]:
    embeddings = np.load(npz_path)["embeddings"].astype(np.float32)
    with metadata_csv.open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    if len(metadata) != len(embeddings):
        raise ValueError(f"{metadata_csv} has {len(metadata)} rows but {npz_path} has {len(embeddings)} embeddings.")
    return embeddings, metadata


def dmso_center(embeddings: np.ndarray, metadata: list[dict[str, str]]) -> np.ndarray:
    """Subtract each plate's mean DMSO embedding (removes plate/batch offsets)."""
    centered = embeddings.copy()
    by_plate: dict[str, list[int]] = defaultdict(list)
    dmso_by_plate: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        by_plate[row["plate"]].append(index)
        if row["compound"] == "DMSO":
            dmso_by_plate[row["plate"]].append(index)

    global_dmso = [i for indices in dmso_by_plate.values() for i in indices]
    global_mean = embeddings[global_dmso].mean(axis=0) if global_dmso else embeddings.mean(axis=0)
    for plate, indices in by_plate.items():
        reference = embeddings[dmso_by_plate[plate]].mean(axis=0) if dmso_by_plate[plate] else global_mean
        centered[indices] -= reference
    return centered


def aggregate_profiles(
    embeddings: np.ndarray,
    metadata: list[dict[str, str]],
    keys: Sequence[str] = ("compound",),
) -> dict[tuple[str, ...], np.ndarray]:
    """Mean embedding per group, e.g. per compound or per (compound, concentration) treatment."""
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[tuple(row[key] for key in keys)].append(index)
    return {key: embeddings[indices].mean(axis=0) for key, indices in groups.items()}


def compound_phenotype_profiles(
    npz_path: Path, metadata_csv: Path, center_on_dmso: bool = True
) -> dict[str, np.ndarray]:
    """One phenotype vector per compound: DMSO-centred mean image embedding."""
    embeddings, metadata = read_embedding_table(npz_path, metadata_csv)
    if center_on_dmso:
        embeddings = dmso_center(embeddings, metadata)
    return {key[0]: profile for key, profile in aggregate_profiles(embeddings, metadata, ("compound",)).items()}
