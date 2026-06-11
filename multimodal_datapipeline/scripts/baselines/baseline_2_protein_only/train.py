#!/usr/bin/env python3
"""Train Baseline 2: protein-only prediction with ESM-2 embeddings.

Default task:
    protein_sequence -> frozen ESM-2 encoder -> MLP -> active_fraction regression

The processed Baseline 2 table has only 12 target-level rows and all binary
labels are currently active, so regression on `active_fraction` is the useful
default. Binary classification is available only when both classes exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from multimodal_datapipeline.models.protein_encoder import ESM2ProteinEncoder  # noqa: E402


DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "baseline_2_protein_only.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "experiments" / "baseline_2_protein_only"


class ProteinDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], target_column: str) -> None:
        self.rows = rows
        self.target_column = target_column

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "sequence": row["protein_sequence"],
            "target": float(row[self.target_column]),
            "target_chembl_id": row["target_chembl_id"],
            "uniprot_id": row["uniprot_id"],
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sequences": [str(item["sequence"]) for item in batch],
        "targets": torch.tensor([float(item["target"]) for item in batch], dtype=torch.float32),
        "target_chembl_ids": [str(item["target_chembl_id"]) for item in batch],
        "uniprot_ids": [str(item["uniprot_id"]) for item in batch],
    }


class ProteinOnlyPredictor(nn.Module):
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        freeze_backbone: bool,
        max_length: int,
    ) -> None:
        super().__init__()
        self.encoder = ESM2ProteinEncoder(
            model_name=model_name,
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone,
            max_length=max_length,
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequences: list[str]) -> torch.Tensor:
        embeddings = self.encoder(sequences)
        return self.head(embeddings).squeeze(-1)


def read_rows(path: Path, task: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"target_chembl_id", "uniprot_id", "protein_sequence", "active_fraction", "label"}
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    target_column = "active_fraction" if task == "regression" else "label"
    usable = [row for row in rows if row["protein_sequence"] and row[target_column] != ""]
    if len(usable) < 4:
        raise ValueError(f"Need at least 4 usable protein rows, found {len(usable)}")
    if task == "classification" and len({row["label"] for row in usable}) < 2:
        raise ValueError("Classification needs both classes, but Baseline 2 currently has one class.")
    return usable


def split_rows(rows: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, round(n_total * 0.2))
    n_val = max(1, round(n_total * 0.2))
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict[str, float]:
    n = len(y_true)
    errors = [pred - true for true, pred in zip(y_true, y_pred)]
    mse = sum(error * error for error in errors) / n
    mae = sum(abs(error) for error in errors) / n
    mean_true = sum(y_true) / n
    ss_tot = sum((true - mean_true) ** 2 for true in y_true)
    ss_res = sum(error * error for error in errors)
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": mae, "r2": r2}


def classification_metrics(y_true: list[int], logits: list[float]) -> dict[str, float]:
    probs = [1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit)))) for logit in logits]
    preds = [int(prob >= 0.5) for prob in probs]
    tp = sum(true == 1 and pred == 1 for true, pred in zip(y_true, preds))
    tn = sum(true == 0 and pred == 0 for true, pred in zip(y_true, preds))
    fp = sum(true == 0 and pred == 1 for true, pred in zip(y_true, preds))
    fn = sum(true == 1 and pred == 0 for true, pred in zip(y_true, preds))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    output_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            logits = model(batch["sequences"])
            predictions = torch.sigmoid(logits) if task == "regression" else logits
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            for target_id, uniprot_id, target, pred in zip(
                batch["target_chembl_ids"],
                batch["uniprot_ids"],
                targets.cpu().tolist(),
                predictions.cpu().tolist(),
            ):
                output_rows.append(
                    {
                        "target_chembl_id": target_id,
                        "uniprot_id": uniprot_id,
                        "target": target,
                        "prediction": pred,
                    }
                )

    if task == "regression":
        return regression_metrics(y_true, y_pred), output_rows
    return classification_metrics([int(value) for value in y_true], y_pred), output_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 2 protein-only model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fine-tune-backbone", action="store_true")
    return parser


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.data, args.task)
    target_column = "active_fraction" if args.task == "regression" else "label"
    splits = split_rows(rows, args.seed)

    device = select_device(args.device)
    try:
        model = ProteinOnlyPredictor(
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            freeze_backbone=not args.fine_tune_backbone,
            max_length=args.max_length,
        ).to(device)
    except OSError as exc:
        raise SystemExit(
            "Could not load the ESM-2 pretrained model. The first run needs access to "
            "Hugging Face to download `facebook/esm2_t6_8M_UR50D`, or you must pass "
            "`--model-name /path/to/local/esm2_model_directory`.\n"
            f"Original error: {exc}"
        ) from exc

    train_loader = DataLoader(
        ProteinDataset(splits["train"], target_column),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ProteinDataset(splits["val"], target_column),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        ProteinDataset(splits["test"], target_column),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn: nn.Module = nn.MSELoss() if args.task == "regression" else nn.BCEWithLogitsLoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            targets = batch["targets"].to(device)
            if args.task == "regression":
                targets = targets.clamp(0.0, 1.0)
                outputs = torch.sigmoid(model(batch["sequences"]))
            else:
                outputs = model(batch["sequences"])

            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val_loader, args.task, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(losses) / max(1, len(losses)),
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )

    train_metrics, _ = evaluate(model, train_loader, args.task, device)
    val_metrics, _ = evaluate(model, val_loader, args.task, device)
    test_metrics, test_predictions = evaluate(model, test_loader, args.task, device)

    metrics = {
        "task": f"protein_only_{args.task}",
        "input_csv": str(args.data),
        "model_name": args.model_name,
        "target_column": target_column,
        "device": str(device),
        "n_rows": len(rows),
        "split_sizes": {name: len(split_rows_) for name, split_rows_ in splits.items()},
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
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["target_chembl_id", "uniprot_id", "target", "prediction"],
        )
        writer.writeheader()
        writer.writerows(test_predictions)

    print("Protein-only baseline complete")
    print(f"Rows: {len(rows)}")
    print(f"Task: {args.task}")
    print(f"Device: {device}")
    print(f"Test metrics: {test_metrics}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
