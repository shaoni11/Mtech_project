#!/usr/bin/env python3
"""Experiment 1: molecule-only neural baseline.

Task:
    SMILES -> Morgan fingerprint -> MLP -> active / inactive

This script reads the curated ChEMBL molecule table, collapses repeated SMILES
into one molecule-level example, trains a PyTorch MLP, and writes:

- metrics.json
- test_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_PROJECT_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
MULTIMODAL_SRC_DIR = REPO_PROJECT_DIR / "multimodal_datapipeline" / "src"
for path in (SRC_DIR, MULTIMODAL_SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deep_learning_project.metrics import binary_classification_metrics 
from deep_learning_project.splits import stratified_indices
from multimodal_datapipeline.models.molecule_encoder import ( 
    MorganFingerprintConfig,
    MorganFingerprintFeaturizer,
)


DEFAULT_DATA = REPO_PROJECT_DIR / "multimodal_datapipeline" / "data" / "processed" / "chembl_molecule_curated.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "experiments" / "experiment_1_molecule_only_mlp"


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def load_molecule_examples(csv_path: Path, activity_threshold: float) -> list[dict[str, object]]:
    """Collapse repeated SMILES across targets into one molecule-level label."""
    grouped: dict[str, dict[str, object]] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"curated_smiles", "median_pchembl", "target_chembl_id", "molecule_chembl_ids"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

        for row in reader:
            smiles = (row.get("curated_smiles") or "").strip()
            pchembl = parse_float(row.get("median_pchembl"))
            if not smiles or pchembl is None:
                continue

            item = grouped.setdefault(
                smiles,
                {
                    "smiles": smiles,
                    "pchembl_values": [],
                    "target_chembl_ids": set(),
                    "molecule_chembl_ids": set(),
                },
            )
            item["pchembl_values"].append(pchembl)
            item["target_chembl_ids"].add(row.get("target_chembl_id", ""))
            for molecule_id in (row.get("molecule_chembl_ids") or "").split(";"):
                if molecule_id:
                    item["molecule_chembl_ids"].add(molecule_id)

    examples: list[dict[str, object]] = []
    for item in grouped.values():
        values = item["pchembl_values"]
        mean_pchembl = float(sum(values) / len(values))
        examples.append(
            {
                "smiles": item["smiles"],
                "mean_pchembl": mean_pchembl,
                "label": int(mean_pchembl >= activity_threshold),
                "n_targets": len(item["target_chembl_ids"]),
                "target_chembl_ids": sorted(item["target_chembl_ids"]),
                "molecule_chembl_ids": sorted(item["molecule_chembl_ids"]),
            }
        )

    if len({int(example["label"]) for example in examples}) < 2:
        raise ValueError("Need both active and inactive molecules for classification.")
    return examples


class FingerprintDataset(Dataset):
    def __init__(self, fingerprints: np.ndarray, labels: np.ndarray, examples: list[dict[str, object]], indices: list[int]) -> None:
        self.fingerprints = fingerprints
        self.labels = labels
        self.examples = examples
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        index = self.indices[item]
        return {
            "index": index,
            "fingerprint": torch.from_numpy(self.fingerprints[index]).float(),
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
            "example": self.examples[index],
        }

class MoleculeMLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.network(fingerprints).squeeze(-1)


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "indices": [int(item["index"]) for item in batch],
        "fingerprints": torch.stack([item["fingerprint"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "examples": [item["example"] for item in batch],
    }


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def featurize_examples(examples: list[dict[str, object]], n_bits: int, radius: int) -> np.ndarray:
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.warning")
    except ModuleNotFoundError:
        pass

    featurizer = MorganFingerprintFeaturizer(
        MorganFingerprintConfig(radius=radius, n_bits=n_bits, use_chirality=True)
    )
    smiles = [str(example["smiles"]) for example in examples]
    return featurizer.transform(smiles).astype(np.float32)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            fingerprints = batch["fingerprints"].to(device)
            labels = batch["labels"].to(device)
            logits = model(fingerprints)
            probs = torch.sigmoid(logits)

            label_values = [int(value) for value in labels.cpu().tolist()]
            prob_values = [float(value) for value in probs.cpu().tolist()]
            y_true.extend(label_values)
            y_score.extend(prob_values)

            for example, label, prob in zip(batch["examples"], label_values, prob_values):
                predictions.append(
                    {
                        "smiles": example["smiles"],
                        "label": label,
                        "prob_active": prob,
                        "pred_label": int(prob >= 0.5),
                        "mean_pchembl": example["mean_pchembl"],
                        "n_targets": example["n_targets"],
                        "target_chembl_ids": ";".join(example["target_chembl_ids"]),
                        "molecule_chembl_ids": ";".join(example["molecule_chembl_ids"]),
                    }
                )

    return binary_classification_metrics(y_true, y_score), predictions


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_molecule_examples(args.data, args.activity_threshold)
    labels = np.array([int(example["label"]) for example in examples], dtype=np.int64)
    fingerprints = featurize_examples(examples, args.n_bits, args.radius)

    train_idx, val_idx, test_idx = stratified_indices(
        labels.tolist(),
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        FingerprintDataset(fingerprints, labels, examples, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        FingerprintDataset(fingerprints, labels, examples, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        FingerprintDataset(fingerprints, labels, examples, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = select_device(args.device)
    model = MoleculeMLPClassifier(
        input_dim=args.n_bits,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
    ).to(device)

    n_pos = float(labels[train_idx].sum())
    n_neg = float(len(train_idx) - labels[train_idx].sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_auc = -float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            fingerprints_batch = batch["fingerprints"].to(device)
            labels_batch = batch["labels"].to(device)
            logits = model(fingerprints_batch)
            loss = loss_fn(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val_loader, device)
        val_auc = val_metrics["roc_auc"]
        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": sum(losses) / max(1, len(losses)),
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_f1": val_metrics["f1"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, _ = evaluate(model, train_loader, device)
    val_metrics, _ = evaluate(model, val_loader, device)
    test_metrics, test_predictions = evaluate(model, test_loader, device)

    metrics: dict[str, object] = {
        "experiment": "experiment_1_molecule_only_mlp",
        "task": "smiles_to_active_inactive",
        "input_csv": str(args.data),
        "feature": {
            "type": "rdkit_morgan_fingerprint",
            "n_bits": args.n_bits,
            "radius": args.radius,
            "use_chirality": True,
        },
        "model": {
            "type": "mlp_classifier",
            "hidden_dim": args.hidden_dim,
            "embedding_dim": args.embedding_dim,
            "dropout": args.dropout,
        },
        "activity_threshold_pchembl": args.activity_threshold,
        "device": str(device),
        "n_unique_smiles": len(examples),
        "n_active": int(labels.sum()),
        "n_inactive": int(len(labels) - labels.sum()),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
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
            fieldnames=[
                "smiles",
                "label",
                "prob_active",
                "pred_label",
                "mean_pchembl",
                "n_targets",
                "target_chembl_ids",
                "molecule_chembl_ids",
            ],
        )
        writer.writeheader()
        writer.writerows(test_predictions)

    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "predictions_path": predictions_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 1: molecule-only Morgan fingerprint MLP.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = train(args)
    metrics = result["metrics"]
    print("Experiment 1 complete")
    print(f"Unique SMILES: {metrics['n_unique_smiles']}")
    print(f"Device: {metrics['device']}")
    print(f"Test metrics: {metrics['test']}")
    print(f"Wrote metrics: {result['metrics_path']}")
    print(f"Wrote predictions: {result['predictions_path']}")


if __name__ == "__main__":
    main()
