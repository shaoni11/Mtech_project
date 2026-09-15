#!/usr/bin/env python3
"""Train Baseline 4: molecule + protein activity classification.

Default task:
    curated SMILES -> Morgan fingerprint MLP
    protein sequence -> frozen ESM-2 embedding
    concatenation fusion -> active/inactive

Protein embeddings are computed once per unique UniProt ID and reused across
all molecule-target rows. This keeps the baseline practical on the current
46k-row molecule-protein table.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from multimodal_datapipeline.models.fusion import TwoModalityFusion  # noqa: E402
from multimodal_datapipeline.models.molecule_encoder import (  # noqa: E402
    MoleculeFingerprintMLP,
    MorganFingerprintConfig,
    MorganFingerprintFeaturizer,
)
from multimodal_datapipeline.models.protein_encoder import ESM2ProteinEncoder  # noqa: E402


DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "baseline_4_molecule_protein.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "baseline_4_molecule_protein"


def read_rows(path: Path, max_examples: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "curated_smiles",
        "target_chembl_id",
        "uniprot_id",
        "protein_sequence",
        "median_pchembl",
        "label",
        "molecule_chembl_ids",
    }
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    usable = [
        row
        for row in rows
        if row["curated_smiles"] and row["protein_sequence"] and row["label"] in {"0", "1"}
    ]
    if max_examples is not None:
        usable = stratified_limit(usable, max_examples)
    if len(usable) < 10:
        raise ValueError(f"Need at least 10 usable molecule-protein rows, found {len(usable)}")
    if len({row["label"] for row in usable}) < 2:
        raise ValueError("Need both active and inactive labels for Baseline 4 classification.")
    return usable


def stratified_limit(rows: list[dict[str, str]], max_examples: int) -> list[dict[str, str]]:
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    rng = random.Random(42)
    limited: list[dict[str, str]] = []
    per_label = max(1, max_examples // max(1, len(by_label)))
    for label_rows in by_label.values():
        shuffled = label_rows[:]
        rng.shuffle(shuffled)
        limited.extend(shuffled[:per_label])

    if len(limited) < max_examples:
        limited_ids = {id(row) for row in limited}
        remaining = [row for row in rows if id(row) not in limited_ids]
        rng.shuffle(remaining)
        limited.extend(remaining[: max_examples - len(limited)])
    rng.shuffle(limited)
    return limited[:max_examples]


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


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


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


def pr_auc_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return float("nan")

    order = np.argsort(-probs)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    fp = np.cumsum(sorted_true == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / positives
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    y_pred = (probs >= 0.5).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": pr_auc_score(y_true, probs),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def build_fingerprints(rows: list[dict[str, str]], fingerprint_bits: int, radius: int) -> np.ndarray:
    featurizer = MorganFingerprintFeaturizer(
        MorganFingerprintConfig(radius=radius, n_bits=fingerprint_bits)
    )
    return featurizer.transform([row["curated_smiles"] for row in rows])


def compute_protein_embeddings(
    rows: list[dict[str, str]],
    model_name: str,
    embedding_dim: int,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sequence_by_uniprot: dict[str, str] = {}
    for row in rows:
        sequence_by_uniprot.setdefault(row["uniprot_id"], row["protein_sequence"])

    encoder = ESM2ProteinEncoder(
        model_name=model_name,
        embedding_dim=embedding_dim,
        freeze_backbone=True,
        max_length=max_length,
    ).to(device)
    encoder.eval()

    embeddings: dict[str, torch.Tensor] = {}
    items = list(sequence_by_uniprot.items())
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            uniprot_ids = [item[0] for item in batch]
            sequences = [item[1] for item in batch]
            batch_embeddings = encoder(sequences).detach().cpu()
            for uniprot_id, embedding in zip(uniprot_ids, batch_embeddings):
                embeddings[uniprot_id] = embedding
    return embeddings


class MoleculeProteinDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        fingerprints: np.ndarray,
        protein_embeddings: dict[str, torch.Tensor],
        indices: np.ndarray,
    ) -> None:
        self.rows = rows
        self.fingerprints = fingerprints
        self.protein_embeddings = protein_embeddings
        self.indices = indices.tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_idx = self.indices[index]
        row = self.rows[row_idx]
        return {
            "fingerprint": torch.from_numpy(self.fingerprints[row_idx]),
            "protein_embedding": self.protein_embeddings[row["uniprot_id"]],
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "row_idx": row_idx,
        }


class MoleculeProteinClassifier(nn.Module):
    def __init__(
        self,
        fingerprint_bits: int,
        molecule_hidden_dim: int,
        embedding_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.molecule_encoder = MoleculeFingerprintMLP(
            input_dim=fingerprint_bits,
            hidden_dim=molecule_hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.fusion = TwoModalityFusion.molecule_protein(
            molecule_dim=embedding_dim,
            protein_dim=embedding_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=1,
            dropout=dropout,
        )

    def forward(self, fingerprints: torch.Tensor, protein_embeddings: torch.Tensor) -> torch.Tensor:
        molecule_embeddings = self.molecule_encoder(fingerprints)
        logits = self.fusion(
            molecule_embedding=molecule_embeddings,
            protein_embedding=protein_embeddings,
        )
        return logits.squeeze(-1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[float] = []
    logits_out: list[float] = []
    row_indices: list[int] = []
    with torch.no_grad():
        for batch in loader:
            fingerprints = batch["fingerprint"].to(device)
            protein_embeddings = batch["protein_embedding"].to(device)
            labels = batch["label"].to(device)
            logits = model(fingerprints, protein_embeddings)
            y_true.extend(labels.cpu().tolist())
            logits_out.extend(logits.cpu().tolist())
            row_indices.extend([int(idx) for idx in batch["row_idx"].tolist()])

    y_true_array = np.array(y_true, dtype=np.int32)
    probs = sigmoid(np.array(logits_out, dtype=np.float32))
    metrics = {
        "loss": log_loss(y_true_array.astype(np.float32), probs),
        **classification_metrics(y_true_array, probs),
        "n_examples": int(len(y_true_array)),
        "n_active": int(np.sum(y_true_array == 1)),
        "n_inactive": int(np.sum(y_true_array == 0)),
    }
    predictions = [
        {"row_idx": row_idx, "label": int(label), "predicted_probability_active": float(prob)}
        for row_idx, label, prob in zip(row_indices, y_true_array.tolist(), probs.tolist())
    ]
    return metrics, predictions


def write_predictions(
    path: Path,
    rows: list[dict[str, str]],
    predictions: list[dict[str, object]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "curated_smiles",
                "molecule_chembl_ids",
                "target_chembl_id",
                "uniprot_id",
                "median_pchembl",
                "label",
                "predicted_probability_active",
            ],
        )
        writer.writeheader()
        for prediction in predictions:
            row = rows[int(prediction["row_idx"])]
            writer.writerow(
                {
                    "curated_smiles": row["curated_smiles"],
                    "molecule_chembl_ids": row["molecule_chembl_ids"],
                    "target_chembl_id": row["target_chembl_id"],
                    "uniprot_id": row["uniprot_id"],
                    "median_pchembl": row["median_pchembl"],
                    "label": prediction["label"],
                    "predicted_probability_active": f"{prediction['predicted_probability_active']:.6f}",
                }
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 4 molecule + protein fusion model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--molecule-hidden-dim", type=int, default=512)
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--protein-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-examples", type=int, default=None, help="Optional smoke-test limit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.data, max_examples=args.max_examples)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int32)
    train_idx, val_idx, test_idx = stratified_split(labels, args.test_size, args.val_size, args.seed)

    device = select_device(args.device)
    print(f"Rows: {len(rows)}")
    print(f"Active/inactive: {int(np.sum(labels == 1))}/{int(np.sum(labels == 0))}")
    print(f"Unique proteins: {len({row['uniprot_id'] for row in rows})}")
    print(f"Device: {device}")
    print("Building Morgan fingerprints")
    fingerprints = build_fingerprints(rows, args.fingerprint_bits, args.fingerprint_radius)

    print("Computing frozen protein embeddings")
    try:
        protein_embeddings = compute_protein_embeddings(
            rows=rows,
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            max_length=args.max_length,
            batch_size=args.protein_batch_size,
            device=device,
        )
    except OSError as exc:
        raise SystemExit(
            "Could not load the ESM-2 pretrained model. The first run needs access to "
            "Hugging Face to download `facebook/esm2_t6_8M_UR50D`, or you must pass "
            "`--model-name /path/to/local/esm2_model_directory`.\n"
            f"Original error: {exc}"
        ) from exc

    train_loader = DataLoader(
        MoleculeProteinDataset(rows, fingerprints, protein_embeddings, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MoleculeProteinDataset(rows, fingerprints, protein_embeddings, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MoleculeProteinDataset(rows, fingerprints, protein_embeddings, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = MoleculeProteinClassifier(
        fingerprint_bits=args.fingerprint_bits,
        molecule_hidden_dim=args.molecule_hidden_dim,
        embedding_dim=args.embedding_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            fingerprints_batch = batch["fingerprint"].to(device)
            protein_batch = batch["protein_embedding"].to(device)
            labels_batch = batch["label"].to(device)

            logits = model(fingerprints_batch, protein_batch)
            loss = loss_fn(logits, labels_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val_loader, device)
        epoch_row = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_row)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={epoch_row['train_loss']:.4f} "
            f"val_roc_auc={val_metrics['roc_auc']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f}"
        )

    train_metrics, _ = evaluate(model, train_loader, device)
    val_metrics, _ = evaluate(model, val_loader, device)
    test_metrics, test_predictions = evaluate(model, test_loader, device)

    metrics = {
        "task": "molecule_protein_activity_classification",
        "input_csv": str(args.data),
        "model": "MoleculeFingerprintMLP + frozen ESM2ProteinEncoder + TwoModalityFusion",
        "model_name": args.model_name,
        "device": str(device),
        "n_rows": len(rows),
        "n_active": int(np.sum(labels == 1)),
        "n_inactive": int(np.sum(labels == 0)),
        "n_unique_molecules": len({row["curated_smiles"] for row in rows}),
        "n_unique_targets": len({row["target_chembl_id"] for row in rows}),
        "n_unique_uniprot_ids": len({row["uniprot_id"] for row in rows}),
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
    write_predictions(predictions_path, rows, test_predictions)

    model_path = args.out_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": metrics["args"],
            "test_metrics": test_metrics,
        },
        model_path,
    )

    print("Molecule + protein baseline complete")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote model: {model_path}")


if __name__ == "__main__":
    main()
