#!/usr/bin/env python3
"""Extract frozen DINOv2 embeddings for every BBBC021 image (with or without MoA).

Outputs (used by Baselines 5-7 as the phenotype modality):
    data/interim/bbbc021_dinov2_embeddings.npz     embeddings [n_images, hidden]
    data/interim/bbbc021_image_metadata.csv        one row per embedding
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from multimodal_datapipeline.utils.microscopy import load_three_channel_array  # noqa: E402


BBBC = PROJECT_ROOT / "dataset_pipeline_output" / "bbbc021"
DEFAULT_OUT_NPZ = PROJECT_ROOT / "data" / "interim" / "bbbc021_dinov2_embeddings.npz"
DEFAULT_OUT_CSV = PROJECT_ROOT / "data" / "interim" / "bbbc021_image_metadata.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_image_rows(bbbc_dir: Path) -> list[dict[str, str]]:
    files = {path.name: str(path) for path in (bbbc_dir / "images").rglob("*.tif")}
    smiles = {row["compound"]: row.get("smiles", "") for row in read_csv(bbbc_dir / "BBBC021_v1_compound.csv")}
    moa = {(row["compound"], row["concentration"]): row["moa"] for row in read_csv(bbbc_dir / "BBBC021_v1_moa.csv")}

    rows = []
    for row in read_csv(bbbc_dir / "BBBC021_v1_image.csv"):
        paths = [files.get(row[f"Image_FileName_{channel}"]) for channel in ("DAPI", "Tubulin", "Actin")]
        if not all(paths):
            continue
        compound = row["Image_Metadata_Compound"]
        concentration = row["Image_Metadata_Concentration"]
        plate = row["Image_Metadata_Plate_DAPI"]
        rows.append(
            {
                "compound": compound,
                "smiles": smiles.get(compound, ""),
                "concentration": concentration,
                "moa": moa.get((compound, concentration), ""),
                "plate": plate,
                "week": plate.split("_", 1)[0],
                "well": row["Image_Metadata_Well_DAPI"],
                "replicate": row["Replicate"],
                "dapi_path": paths[0],
                "tubulin_path": paths[1],
                "actin_path": paths[2],
            }
        )
    return rows


class ImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], image_size: int) -> None:
        self.rows = rows
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.rows[index]
        paths = [row["dapi_path"], row["tubulin_path"], row["actin_path"]]
        return torch.from_numpy(load_three_channel_array(paths, self.image_size))


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract DINOv2 embeddings for all BBBC021 images.")
    parser.add_argument("--bbbc-dir", type=Path, default=BBBC)
    parser.add_argument("--out-npz", type=Path, default=DEFAULT_OUT_NPZ)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from transformers import AutoModel

    rows = build_image_rows(args.bbbc_dir)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Images with all three channels: {len(rows)}", flush=True)

    device = select_device(args.device)
    backbone = AutoModel.from_pretrained(args.model_name).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    loader = DataLoader(
        ImageDataset(rows, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    chunks = []
    with torch.no_grad():
        for batch_index, images in enumerate(loader):
            pixel_values = F.interpolate(images.to(device), size=(224, 224), mode="bilinear", align_corners=False)
            outputs = backbone(pixel_values=(pixel_values - mean) / std)
            pooled = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else (
                outputs.last_hidden_state[:, 0, :]
            )
            chunks.append(pooled.float().cpu().numpy())
            if batch_index % 50 == 0:
                print(f"  {min(len(rows), (batch_index + 1) * args.batch_size)}/{len(rows)}", flush=True)

    embeddings = np.concatenate(chunks, axis=0)
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, embeddings=embeddings)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "images": len(rows),
        "embedding_dim": int(embeddings.shape[1]),
        "compounds": len({row["compound"] for row in rows}),
        "images_with_moa": sum(bool(row["moa"]) for row in rows),
        "model_name": args.model_name,
        "out_npz": str(args.out_npz),
        "out_csv": str(args.out_csv),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
