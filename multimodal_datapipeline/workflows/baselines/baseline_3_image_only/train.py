#!/usr/bin/env python3
"""Train Baseline 3: BBBC021 image-only MoA classification.

The processed table is produced by:
workflows/baselines/baseline_3_image_only/curate_data.py

Each example contains three microscopy channels:
- DAPI
- Tubulin
- Actin

This baseline intentionally uses only image inputs and predicts the mechanism of
action label from BBBC021.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "baseline_3_image_only.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "baseline_3_image_only"


CHANNEL_COLUMNS = ["dapi_path", "tubulin_path", "actin_path"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "compound",
        "concentration",
        "moa",
        "plate",
        "well",
        "replicate",
        *CHANNEL_COLUMNS,
    }
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    usable = [
        row
        for row in rows
        if row["moa"] and all(row.get(column) and Path(row[column]).exists() for column in CHANNEL_COLUMNS)
    ]
    if len(usable) < 10:
        raise ValueError(f"Need at least 10 usable image rows, found {len(usable)}")
    if len({row["moa"] for row in usable}) < 2:
        raise ValueError("Image-only classification needs at least two MoA classes.")
    return usable


def make_label_maps(rows: list[dict[str, str]]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({row["moa"] for row in rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def stratified_split(
    rows: list[dict[str, str]],
    label_to_id: dict[str, int],
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[label_to_id[row["moa"]]].append(row)

    splits = {"train": [], "val": [], "test": []}
    for class_rows in by_label.values():
        shuffled = class_rows[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = max(1, round(n_total * test_size))
        n_val = max(1, round(n_total * val_size))
        if n_total - n_test - n_val < 1:
            n_test = 1
            n_val = 0 if n_total < 3 else 1

        splits["test"].extend(shuffled[:n_test])
        splits["val"].extend(shuffled[n_test : n_test + n_val])
        splits["train"].extend(shuffled[n_test + n_val :])

    for split_rows in splits.values():
        rng.shuffle(split_rows)
    return splits


def load_channel(path: str, image_size: int) -> np.ndarray:
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


def load_three_channel_image(row: dict[str, str], image_size: int) -> torch.Tensor:
    channels = [load_channel(row[column], image_size) for column in CHANNEL_COLUMNS]
    image = np.stack(channels, axis=0)
    return torch.from_numpy(image)


class BBBC021ImageDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        label_to_id: dict[str, int],
        image_size: int,
        augment: bool = False,
        cache_images: bool = False,
    ) -> None:
        self.rows = rows
        self.label_to_id = label_to_id
        self.image_size = image_size
        self.augment = augment
        self.cached_images = None
        if cache_images:
            self.cached_images = [load_three_channel_image(row, image_size) for row in rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        if self.cached_images is None:
            image = load_three_channel_image(row, self.image_size)
        else:
            image = self.cached_images[index]
        if self.augment:
            image = augment_image(image)
        return {
            "image": image,
            "label": self.label_to_id[row["moa"]],
            "moa": row["moa"],
            "compound": row["compound"],
            "concentration": row["concentration"],
            "plate": row["plate"],
            "well": row["well"],
            "replicate": row["replicate"],
        }


def augment_image(image: torch.Tensor) -> torch.Tensor:
    """Lightweight microscopy-safe augmentation for small data experiments."""
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=[1])
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=[2])
    if torch.rand(()) < 0.5:
        k = int(torch.randint(1, 4, ()).item())
        image = torch.rot90(image, k=k, dims=[1, 2])

    brightness = 0.9 + 0.2 * torch.rand(1, 1, 1)
    noise = torch.randn_like(image) * 0.015
    image = image * brightness + noise
    return image.clamp(0.0, 1.0)


class TinyMicroscopyCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images.float()))


class SmallMicroscopyCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images.float()))


class DINOv2LinearClassifier(nn.Module):
    """Frozen DINOv2 backbone with a trainable MoA classifier head."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "facebook/dinov2-small",
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DINOv2 experiments require transformers. Install project requirements first."
            ) from exc

        self.backbone = AutoModel.from_pretrained(model_name)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        hidden_size = int(self.backbone.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = F.interpolate(
            images.float(),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )
        pixel_values = (pixel_values - self.mean) / self.std
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            if getattr(outputs, "pooler_output", None) is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]
        return self.classifier(features)


def build_model(
    model_name: str,
    num_classes: int,
    dropout: float,
    dinov2_model_name: str,
) -> nn.Module:
    if model_name == "tiny_cnn":
        return TinyMicroscopyCNN(num_classes=num_classes, dropout=dropout)
    if model_name == "small_cnn":
        return SmallMicroscopyCNN(num_classes=num_classes, dropout=dropout)
    if model_name == "dinov2_linear":
        return DINOv2LinearClassifier(
            num_classes=num_classes,
            model_name=dinov2_model_name,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model: {model_name}")


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": torch.stack([item["image"] for item in batch]),
        "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.long),
        "rows": batch,
    }


def class_weights(rows: list[dict[str, str]], label_to_id: dict[str, int]) -> torch.Tensor:
    counts = Counter(label_to_id[row["moa"]] for row in rows)
    n_classes = len(label_to_id)
    total = sum(counts.values())
    weights = [total / max(1, n_classes * counts[class_id]) for class_id in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> dict[str, float]:
    total = len(y_true)
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    per_class_f1 = []
    per_class_recall = []
    per_class_precision = []

    for class_id in range(num_classes):
        tp = sum(true == class_id and pred == class_id for true, pred in zip(y_true, y_pred))
        fp = sum(true != class_id and pred == class_id for true, pred in zip(y_true, y_pred))
        fn = sum(true == class_id and pred != class_id for true, pred in zip(y_true, y_pred))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": correct / max(1, total),
        "macro_precision": float(sum(per_class_precision) / num_classes),
        "macro_recall": float(sum(per_class_recall) / num_classes),
        "macro_f1": float(sum(per_class_f1) / num_classes),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    id_to_label: dict[int, str],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    losses = []
    y_true: list[int] = []
    y_pred: list[int] = []
    output_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            losses.append(float(loss.detach().cpu()))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

            for item, pred_id, true_id, confidence in zip(
                batch["rows"],
                preds.cpu().tolist(),
                labels.cpu().tolist(),
                probs.max(dim=-1).values.cpu().tolist(),
            ):
                output_rows.append(
                    {
                        "compound": item["compound"],
                        "concentration": item["concentration"],
                        "plate": item["plate"],
                        "well": item["well"],
                        "replicate": item["replicate"],
                        "true_moa": id_to_label[int(true_id)],
                        "predicted_moa": id_to_label[int(pred_id)],
                        "confidence": float(confidence),
                    }
                )

    metrics = classification_metrics(y_true, y_pred, len(id_to_label))
    metrics["loss"] = sum(losses) / max(1, len(losses))
    return metrics, output_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 3 image-only MoA classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--model", default="small_cnn", choices=["tiny_cnn", "small_cnn", "dinov2_linear"])
    parser.add_argument("--dinov2-model-name", default="facebook/dinov2-small")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "compound",
                "concentration",
                "plate",
                "well",
                "replicate",
                "true_moa",
                "predicted_moa",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.data)
    label_to_id, id_to_label = make_label_maps(rows)
    splits = stratified_split(rows, label_to_id, args.test_size, args.val_size, args.seed)

    device = select_device(args.device)
    model = build_model(
        model_name=args.model,
        num_classes=len(label_to_id),
        dropout=args.dropout,
        dinov2_model_name=args.dinov2_model_name,
    ).to(device)

    loaders = {
        name: DataLoader(
            BBBC021ImageDataset(
                split_rows,
                label_to_id,
                args.image_size,
                augment=args.augment and name == "train",
                cache_images=args.cache_images,
            ),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            collate_fn=collate_batch,
        )
        for name, split_rows in splits.items()
    }

    weights = class_weights(splits["train"], label_to_id).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in loaders["train"]:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, loaders["val"], loss_fn, device, id_to_label)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(1, len(train_losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_metrics)
        if args.log_every and (epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0):
            print(
                "Epoch "
                f"{epoch:03d}/{args.epochs} "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"val_acc={epoch_metrics['val_accuracy']:.4f} "
                f"val_macro_f1={epoch_metrics['val_macro_f1']:.4f}",
                flush=True,
            )

    train_metrics, _ = evaluate(model, loaders["train"], loss_fn, device, id_to_label)
    val_metrics, _ = evaluate(model, loaders["val"], loss_fn, device, id_to_label)
    test_metrics, test_predictions = evaluate(model, loaders["test"], loss_fn, device, id_to_label)

    metrics = {
        "task": "image_only_moa_classification",
        "input_csv": str(args.data),
        "model": args.model,
        "dinov2_model_name": args.dinov2_model_name if args.model == "dinov2_linear" else None,
        "channels": CHANNEL_COLUMNS,
        "image_size": args.image_size,
        "augment": args.augment,
        "cache_images": args.cache_images,
        "device": str(device),
        "n_rows": len(rows),
        "labels": id_to_label,
        "class_counts": dict(Counter(row["moa"] for row in rows)),
        "split_sizes": {name: len(split_rows) for name, split_rows in splits.items()},
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "args": vars(args) | {"data": str(args.data), "out_dir": str(args.out_dir)},
    }

    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    predictions_path = args.out_dir / "test_predictions.csv"
    write_predictions(predictions_path, test_predictions)

    if args.save_model:
        torch.save(
            {
                "model": model.state_dict(),
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "args": vars(args) | {"data": str(args.data), "out_dir": str(args.out_dir)},
            },
            args.out_dir / "model.pt",
        )

    print("Image-only baseline complete")
    print(f"Rows: {len(rows)}")
    print(f"Classes: {len(label_to_id)}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Test metrics: {test_metrics}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
