#!/usr/bin/env python3
"""Molecule-only baseline for ChEMBL activity classification.

This baseline intentionally uses only SMILES strings. It featurizes each SMILES
with hashed character n-grams and trains a small logistic-regression classifier
with NumPy SGD. No protein, image, or target-structure information is used.

By default it reads the curated molecule table produced by:
scripts/baselines/baseline_1_molecule_only/curate_data.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


DEFAULT_DATA = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "chembl_molecule_curated.csv"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "experiments" / "molecule_only_baseline"


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def sigmoid(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0)))


def featurize_smiles(smiles: str, dim: int, min_n: int = 2, max_n: int = 4) -> np.ndarray:
    """Return L2-normalized hashed character n-gram features for one SMILES."""
    x = np.zeros(dim, dtype=np.float32)
    padded = f"^{smiles}$"
    for n in range(min_n, max_n + 1):
        if len(padded) < n:
            continue
        for i in range(len(padded) - n + 1):
            gram = padded[i : i + n]
            h = stable_hash(gram)
            sign = 1.0 if h % 2 == 0 else -1.0
            x[(h // 2) % dim] += sign

    norm = float(np.linalg.norm(x))
    if norm > 0.0:
        x /= norm
    return x


def parse_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def load_examples(
    csv_path: Path,
    activity_threshold: float,
    target: str | None,
    require_exact_relation: bool,
) -> list[dict[str, object]]:
    """Load rows and collapse repeated SMILES to one molecule-only label.

    Supports both:
    - curated table: data/processed/chembl_molecule_curated.csv
    - raw ChEMBL table: dataset_pipeline_output/chembl/activities_multitarget.csv
    """
    grouped: dict[str, dict[str, object]] = {}

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        is_curated = "curated_smiles" in fieldnames
        if is_curated:
            required = {
                "curated_smiles",
                "median_pchembl",
                "target_chembl_id",
                "molecule_chembl_ids",
                "n_activity_rows",
            }
        else:
            required = {"canonical_smiles", "pchembl_value", "target_chembl_id", "molecule_chembl_id"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

        for row in reader:
            if target and row.get("target_chembl_id") != target:
                continue
            if not is_curated and require_exact_relation and row.get("standard_relation") not in {"=", ""}:
                continue

            if is_curated:
                smiles = (row.get("curated_smiles") or "").strip()
                pchembl = parse_float(row.get("median_pchembl", ""))
                molecule_ids = [
                    molecule_id
                    for molecule_id in (row.get("molecule_chembl_ids") or "").split(";")
                    if molecule_id
                ]
                n_activity_rows = int(row.get("n_activity_rows") or 1)
            else:
                smiles = (row.get("canonical_smiles") or "").strip()
                pchembl = parse_float(row.get("pchembl_value", ""))
                molecule_ids = [row.get("molecule_chembl_id", "")]
                n_activity_rows = 1
            if not smiles or pchembl is None:
                continue

            item = grouped.setdefault(
                smiles,
                {
                    "canonical_smiles": smiles,
                    "molecule_chembl_ids": set(),
                    "target_chembl_ids": set(),
                    "pchembl_values": [],
                    "n_activity_rows": 0,
                },
            )
            item["molecule_chembl_ids"].update(molecule_ids)
            item["target_chembl_ids"].add(row.get("target_chembl_id", ""))
            item["pchembl_values"].append(pchembl)
            item["n_activity_rows"] += n_activity_rows

    examples = []
    for item in grouped.values():
        values = item["pchembl_values"]
        mean_pchembl = float(sum(values) / len(values))
        examples.append(
            {
                "canonical_smiles": item["canonical_smiles"],
                "molecule_chembl_ids": sorted(item["molecule_chembl_ids"]),
                "target_chembl_ids": sorted(item["target_chembl_ids"]),
                "mean_pchembl": mean_pchembl,
                "n_molecule_target_rows": len(values),
                "n_activity_rows": item["n_activity_rows"],
                "label": int(mean_pchembl >= activity_threshold),
            }
        )

    return examples


def stratified_split(
    labels: np.ndarray, test_size: float, val_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels.tolist()):
        by_label[int(label)].append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for indices in by_label.values():
        rng.shuffle(indices)
        n_total = len(indices)
        n_test = max(1, round(n_total * test_size))
        n_val = max(1, round(n_total * val_size))
        test_idx.extend(indices[:n_test])
        val_idx.extend(indices[n_test : n_test + n_val])
        train_idx.extend(indices[n_test + n_val :])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=x_train.shape[1]).astype(np.float32)
    bias = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]

            probs = sigmoid(xb @ weights + bias)
            error = probs - yb
            grad_w = (xb.T @ error) / len(batch_idx) + l2 * weights
            grad_b = float(np.mean(error))

            weights -= lr * grad_w
            bias -= lr * grad_b

        val_probs = sigmoid(x_val @ weights + bias)
        val_metrics = classification_metrics(y_val, val_probs)
        history.append(
            {
                "epoch": float(epoch),
                "val_loss": log_loss(y_val, val_probs),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
            }
        )

    return weights, bias, history


def log_loss(y_true: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    loss = -(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped))
    return float(np.mean(loss))


def roc_auc_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_pred = (probs >= 0.5).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc_score(y_true, probs),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def write_predictions(path: Path, examples: list[dict[str, object]], indices: np.ndarray, probs: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "canonical_smiles",
                "molecule_chembl_ids",
                "target_chembl_ids",
                "mean_pchembl",
                "n_molecule_target_rows",
                "n_activity_rows",
                "label",
                "predicted_probability_active",
            ],
        )
        writer.writeheader()
        for row_idx, prob in zip(indices.tolist(), probs.tolist()):
            example = examples[row_idx]
            writer.writerow(
                {
                    "canonical_smiles": example["canonical_smiles"],
                    "molecule_chembl_ids": ";".join(example["molecule_chembl_ids"]),
                    "target_chembl_ids": ";".join(example["target_chembl_ids"]),
                    "mean_pchembl": f"{example['mean_pchembl']:.4f}",
                    "n_molecule_target_rows": example["n_molecule_target_rows"],
                    "n_activity_rows": example["n_activity_rows"],
                    "label": example["label"],
                    "predicted_probability_active": f"{prob:.6f}",
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a molecule-only SMILES baseline.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Curated molecule CSV path. Raw ChEMBL activity CSVs are also supported.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    parser.add_argument("--target", default=None, help="Optional target_chembl_id filter, e.g. CHEMBL203.")
    parser.add_argument("--activity-threshold", type=float, default=6.0, help="Active if mean pChEMBL >= threshold.")
    parser.add_argument("--feature-dim", type=int, default=2048, help="Hashed SMILES feature dimension.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-inequality-relations",
        action="store_true",
        help="Include rows where standard_relation is not '='.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(
        args.data,
        activity_threshold=args.activity_threshold,
        target=args.target,
        require_exact_relation=not args.allow_inequality_relations,
    )
    if len(examples) < 10:
        raise ValueError(f"Not enough usable molecule examples after filtering: {len(examples)}")

    labels = np.array([example["label"] for example in examples], dtype=np.int32)
    if len(set(labels.tolist())) < 2:
        raise ValueError("Need both active and inactive molecules. Try another threshold or target filter.")

    features = np.stack(
        [featurize_smiles(str(example["canonical_smiles"]), args.feature_dim) for example in examples]
    )

    train_idx, val_idx, test_idx = stratified_split(labels, args.test_size, args.val_size, args.seed)
    weights, bias, history = train_logistic_regression(
        features[train_idx],
        labels[train_idx].astype(np.float32),
        features[val_idx],
        labels[val_idx].astype(np.float32),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
    )

    split_metrics = {}
    for split_name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        probs = sigmoid(features[indices] @ weights + bias)
        split_metrics[split_name] = {
            "loss": log_loss(labels[indices].astype(np.float32), probs),
            **classification_metrics(labels[indices], probs),
            "n_examples": int(len(indices)),
            "n_active": int(np.sum(labels[indices] == 1)),
            "n_inactive": int(np.sum(labels[indices] == 0)),
        }

    metrics = {
        "task": "molecule_only_activity_classification",
        "input_csv": str(args.data),
        "input_type": "curated" if "curated_smiles" in set(csv.DictReader(args.data.open()).fieldnames or []) else "raw",
        "target_filter": args.target,
        "activity_threshold_pchembl": args.activity_threshold,
        "feature": f"hashed_smiles_char_ngrams_dim_{args.feature_dim}",
        "model": "logistic_regression_sgd",
        "n_unique_smiles": len(examples),
        "n_active": int(np.sum(labels == 1)),
        "n_inactive": int(np.sum(labels == 0)),
        "splits": split_metrics,
        "training_history": history,
        "args": vars(args) | {"data": str(args.data), "out_dir": str(args.out_dir)},
    }

    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    test_probs = sigmoid(features[test_idx] @ weights + bias)
    predictions_path = args.out_dir / "test_predictions.csv"
    write_predictions(predictions_path, examples, test_idx, test_probs)

    print("Molecule-only baseline complete")
    print(f"Examples: {len(examples)} unique SMILES")
    print(f"Active/inactive: {metrics['n_active']}/{metrics['n_inactive']}")
    print(f"Test accuracy: {split_metrics['test']['accuracy']:.4f}")
    print(f"Test F1: {split_metrics['test']['f1']:.4f}")
    print(f"Test ROC-AUC: {split_metrics['test']['roc_auc']:.4f}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
