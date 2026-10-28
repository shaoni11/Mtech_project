#!/usr/bin/env python3
"""Molecule-only baseline for ChEMBL activity classification.

This baseline intentionally uses only SMILES strings. No protein, image, or
target-structure information is used.

Models (--model):
    ngram_logreg     hashed character n-grams -> NumPy logistic regression (original baseline)
    fingerprint_mlp  Morgan fingerprint MLP -> fusion head with only the molecule branch
    gnn              molecular graph GNN -> fusion head with only the molecule branch

The fingerprint_mlp and gnn models use the same molecule branch and head as
Baseline 4. Note that this baseline averages each molecule's label over all
targets; for a same-rows ablation of the protein branch use Baseline 4 with
``--protein-encoder none``.

Splits (--split):
    random    stratified random molecules (optimistic)
    scaffold  Bemis-Murcko scaffold groups; no test scaffold is seen in training

By default it reads the curated molecule table produced by:
workflows/baselines/baseline_1_molecule_only/curate_data.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from multimodal_datapipeline.utils.metrics import (  # noqa: E402
    binary_classification_metrics as classification_metrics,
    log_loss,
    sigmoid,
)
from multimodal_datapipeline.utils.splits import (  # noqa: E402
    group_split,
    scaffold_groups,
    split_overlap,
    stratified_random_split,
)


DEFAULT_DATA = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "chembl_molecule_curated.csv"
)
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[3] / "results" / "molecule_only_baseline"


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


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


def make_split(
    examples: list[dict[str, object]], labels: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    smiles = [str(example["canonical_smiles"]) for example in examples]
    scaffolds = scaffold_groups(smiles)
    if args.split == "random":
        train_idx, val_idx, test_idx = stratified_random_split(
            labels.tolist(), args.test_size, args.val_size, args.seed
        )
    elif args.split == "scaffold":
        train_idx, val_idx, test_idx = group_split(scaffolds, args.test_size, args.val_size, args.seed)
    else:
        raise ValueError(f"Unknown split: {args.split}")
    diagnostics = {
        "strategy": args.split,
        "scaffolds": split_overlap(scaffolds, train_idx, val_idx, test_idx),
    }
    return train_idx, val_idx, test_idx, diagnostics


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


def select_device(name: str):
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_torch_model(
    examples: list[dict[str, object]],
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]], str]:
    """Train fingerprint_mlp or gnn; return predicted probabilities per split."""
    import torch
    from torch import nn

    from multimodal_datapipeline.models.fusion import ConcatenationFusion
    from multimodal_datapipeline.models.graph import batch_graphs
    from multimodal_datapipeline.models.molecule_encoder import (
        MoleculeFingerprintMLP,
        MoleculeGNNEncoder,
        MoleculeGraphFeaturizer,
        MorganFingerprintConfig,
        MorganFingerprintFeaturizer,
    )

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    smiles = [str(example["canonical_smiles"]) for example in examples]

    if args.model == "fingerprint_mlp":
        featurizer = MorganFingerprintFeaturizer(MorganFingerprintConfig(n_bits=args.feature_dim))
        fingerprints = torch.from_numpy(featurizer.transform(smiles))
        encoder = MoleculeFingerprintMLP(
            input_dim=args.feature_dim,
            hidden_dim=args.molecule_hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
        )

        def batch_inputs(indices: np.ndarray):
            return fingerprints[torch.from_numpy(indices)].to(device)

    else:
        featurizer = MoleculeGraphFeaturizer()
        graphs = featurizer.transform(smiles)
        encoder = MoleculeGNNEncoder(
            hidden_dim=args.gnn_hidden_dim,
            embedding_dim=args.embedding_dim,
            num_layers=args.gnn_layers,
            dropout=args.dropout,
        )

        def batch_inputs(indices: np.ndarray):
            return batch_graphs([graphs[i] for i in indices.tolist()]).to(device)

    class MoleculeOnlyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.molecule_encoder = encoder
            self.fusion = ConcatenationFusion(
                molecule_dim=args.embedding_dim,
                hidden_dim=args.fusion_hidden_dim,
                output_dim=1,
                dropout=args.dropout,
                use_protein=False,
                use_image=False,
            )

        def forward(self, molecule_input) -> torch.Tensor:
            return self.fusion(molecule_embedding=self.molecule_encoder(molecule_input)).squeeze(-1)

    model = MoleculeOnlyClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)
    loss_fn = nn.BCEWithLogitsLoss()
    label_tensor = torch.from_numpy(labels.astype(np.float32))

    def predict(indices: np.ndarray) -> np.ndarray:
        model.eval()
        logits = []
        with torch.no_grad():
            for start in range(0, len(indices), args.batch_size):
                logits.append(model(batch_inputs(indices[start : start + args.batch_size])).cpu())
        return sigmoid(torch.cat(logits).numpy())

    from multimodal_datapipeline.utils.training import BestEpochTracker

    tracker = BestEpochTracker(patience=args.patience)
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = splits["train"][rng.permutation(len(splits["train"]))]
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            loss = loss_fn(model(batch_inputs(batch_idx)), label_tensor[batch_idx].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_probs = predict(splits["val"])
        val_metrics = classification_metrics(labels[splits["val"]], val_probs)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "val_loss": log_loss(labels[splits["val"]].astype(np.float32), val_probs),
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_pr_auc": val_metrics["pr_auc"],
            }
        )
        improved = tracker.update(epoch, val_metrics["roc_auc"], model)
        print(
            f"Epoch {epoch}/{args.epochs} train_loss={history[-1]['train_loss']:.4f} "
            f"val_roc_auc={val_metrics['roc_auc']:.4f} val_pr_auc={val_metrics['pr_auc']:.4f}"
            + (" *" if improved else ""),
            flush=True,
        )
        if tracker.should_stop:
            print(f"Early stopping: no val ROC-AUC gain for {args.patience} epochs", flush=True)
            break

    tracker.restore(model)
    print(f"Restored best epoch {tracker.best_epoch} (val ROC-AUC {tracker.best_score:.4f})", flush=True)
    return {name: predict(indices) for name, indices in splits.items()}, history, str(device)


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
    parser.add_argument("--model", default="ngram_logreg", choices=["ngram_logreg", "fingerprint_mlp", "gnn"])
    parser.add_argument("--split", default="random", choices=["random", "scaffold"])
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=2048,
        help="Hashed SMILES feature dimension (ngram_logreg) or Morgan fingerprint bits (fingerprint_mlp).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Defaults to 0.2 for ngram_logreg (SGD) and 1e-3 for fingerprint_mlp/gnn (AdamW).",
    )
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 penalty / AdamW weight decay.")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--molecule-hidden-dim", type=int, default=512)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=4)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="fingerprint_mlp/gnn: stop after this many epochs without val ROC-AUC gain (0 = off). "
        "The best validation epoch is always restored.",
    )
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

    if args.learning_rate is None:
        args.learning_rate = 0.2 if args.model == "ngram_logreg" else 1e-3

    train_idx, val_idx, test_idx, split_diagnostics = make_split(examples, labels, args)
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    print(f"Split: {args.split} train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Test scaffolds seen in train: {split_diagnostics['scaffolds']['test_seen_in_train']}")

    device = "cpu"
    if args.model == "ngram_logreg":
        features = np.stack(
            [featurize_smiles(str(example["canonical_smiles"]), args.feature_dim) for example in examples]
        )
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
        split_probs = {name: sigmoid(features[indices] @ weights + bias) for name, indices in splits.items()}
        feature_name = f"hashed_smiles_char_ngrams_dim_{args.feature_dim}"
        model_name = "logistic_regression_sgd"
    else:
        split_probs, history, device = train_torch_model(examples, labels, splits, args)
        feature_name = (
            f"morgan_fingerprint_r2_{args.feature_dim}_bits" if args.model == "fingerprint_mlp" else "molecular_graph"
        )
        model_name = f"{args.model}_molecule_encoder + molecule-only ConcatenationFusion head"

    split_metrics = {}
    for split_name, indices in splits.items():
        probs = split_probs[split_name]
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
        "feature": feature_name,
        "model": model_name,
        "device": device,
        "n_unique_smiles": len(examples),
        "n_active": int(np.sum(labels == 1)),
        "n_inactive": int(np.sum(labels == 0)),
        "split": split_diagnostics,
        "splits": split_metrics,
        "training_history": history,
        "args": vars(args) | {"data": str(args.data), "out_dir": str(args.out_dir)},
    }

    metrics_path = args.out_dir / "metrics.json"
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)

    predictions_path = args.out_dir / "test_predictions.csv"
    write_predictions(predictions_path, examples, test_idx, split_probs["test"])

    print("Molecule-only baseline complete")
    print(f"Examples: {len(examples)} unique SMILES")
    print(f"Active/inactive: {metrics['n_active']}/{metrics['n_inactive']}")
    print(f"Test accuracy: {split_metrics['test']['accuracy']:.4f}")
    print(f"Test F1: {split_metrics['test']['f1']:.4f}")
    print(f"Test ROC-AUC: {split_metrics['test']['roc_auc']:.4f}")
    print(f"Test PR-AUC: {split_metrics['test']['pr_auc']:.4f}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")


if __name__ == "__main__":
    main()
