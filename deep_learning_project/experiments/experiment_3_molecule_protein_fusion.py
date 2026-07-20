#!/usr/bin/env python3
"""Experiment 3: target-aware molecule-protein fusion.
Goal: Does adding protein/target sequence context improve molecule-target activity prediction?
This script is independent of multimodal_datapipeline experiment/model code. It
only uses the shared processed data table as input.

Insight goal:
    Compare a row-level molecule-only model against a molecule+protein fusion
    model on the same molecule-target activity rows.

Models:
    1. molecule_only_pair:
       Morgan fingerprint -> MLP -> active/inactive

    2. molecule_protein_kmer_fusion:
       Morgan fingerprint -> MLP --------\
                                         -> fusion MLP -> active/inactive
       protein sequence k-mer features -> MLP /

Outputs:
    - metrics.json
    - test_predictions.csv
    - comparison_report.md

Interpretation: Adding protein/target context gives a modest but real improvement over molecule-only prediction on scaffold split.    
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_PROJECT_DIR = PROJECT_DIR.parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from deep_learning_utils.featurizers import (  # noqa: E402
    MorganFingerprintConfig,
    MorganFingerprintFeaturizer,
    protein_kmer_feature_matrix,
)
from deep_learning_utils.metrics import binary_classification_metrics  # noqa: E402


DEFAULT_DATA = REPO_PROJECT_DIR / "multimodal_datapipeline" / "data" / "processed" / "baseline_4_molecule_protein.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "experiments" / "experiment_3_molecule_protein_fusion"


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


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "curated_smiles",
            "target_chembl_id",
            "uniprot_id",
            "protein_sequence",
            "median_pchembl",
            "label",
            "molecule_chembl_ids",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

        rows: list[dict[str, object]] = []
        for row in reader:
            smiles = (row.get("curated_smiles") or "").strip()
            sequence = (row.get("protein_sequence") or "").strip()
            pchembl = parse_float(row.get("median_pchembl"))
            label = row.get("label", "")
            if not smiles or not sequence or pchembl is None or label not in {"0", "1"}:
                continue
            rows.append(
                {
                    "smiles": smiles,
                    "target_chembl_id": row["target_chembl_id"],
                    "uniprot_id": row["uniprot_id"],
                    "protein_sequence": sequence,
                    "median_pchembl": pchembl,
                    "label": int(label),
                    "molecule_chembl_ids": row.get("molecule_chembl_ids", ""),
                }
            )

    if len({int(row["label"]) for row in rows}) < 2:
        raise ValueError("Need both active and inactive rows for classification.")
    return rows


def scaffold_for_smiles(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("RDKit is required for scaffold split.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaffold or smiles


def random_stratified_split(labels: list[int], test_size: float, val_size: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for indices in by_label.values():
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test_size))
        n_val = max(1, round(len(indices) * val_size))
        test.extend(indices[:n_test])
        val.extend(indices[n_test : n_test + n_val])
        train.extend(indices[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def scaffold_split(rows: list[dict[str, object]], test_size: float, val_size: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        scaffold_to_indices[scaffold_for_smiles(str(row["smiles"]))].append(index)

    groups = list(scaffold_to_indices.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    n_rows = len(rows)
    target_test = round(n_rows * test_size)
    target_val = round(n_rows * val_size)
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for group in groups:
        if len(test) + len(group) <= target_test:
            test.extend(group)
        elif len(val) + len(group) <= target_val:
            val.extend(group)
        else:
            train.extend(group)

    return train, val, test


def cold_target_split(rows: list[dict[str, object]], seed: int) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    targets = sorted({str(row["target_chembl_id"]) for row in rows})
    rng.shuffle(targets)
    test_targets = set(targets[:2])
    val_targets = set(targets[2:4])
    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for index, row in enumerate(rows):
        target = str(row["target_chembl_id"])
        if target in test_targets:
            test.append(index)
        elif target in val_targets:
            val.append(index)
        else:
            train.append(index)
    return train, val, test


def build_splits(rows: list[dict[str, object]], split: str, test_size: float, val_size: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    labels = [int(row["label"]) for row in rows]
    if split == "random":
        return random_stratified_split(labels, test_size, val_size, seed)
    if split == "scaffold":
        return scaffold_split(rows, test_size, val_size, seed)
    if split == "cold_target":
        return cold_target_split(rows, seed)
    raise ValueError(f"Unsupported split: {split}")


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PairDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, object]],
        molecule_features: np.ndarray,
        protein_features: np.ndarray,
        indices: list[int],
    ) -> None:
        self.rows = rows
        self.molecule_features = molecule_features
        self.protein_features = protein_features
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        index = self.indices[item]
        row = self.rows[index]
        return {
            "index": index,
            "molecule": torch.from_numpy(self.molecule_features[index]).float(),
            "protein": torch.from_numpy(self.protein_features[index]).float(),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "row": row,
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "indices": [int(item["index"]) for item in batch],
        "molecule": torch.stack([item["molecule"] for item in batch]),
        "protein": torch.stack([item["protein"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "rows": [item["row"] for item in batch],
    }


class MoleculeOnlyPairModel(nn.Module):
    def __init__(self, molecule_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(molecule_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, molecule: torch.Tensor, protein: torch.Tensor | None = None) -> torch.Tensor:
        return self.network(molecule).squeeze(-1)


class MoleculeProteinFusionModel(nn.Module):
    def __init__(
        self,
        molecule_dim: int,
        protein_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.molecule_encoder = nn.Sequential(
            nn.Linear(molecule_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, embedding_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, molecule: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        molecule_embedding = self.molecule_encoder(molecule)
        protein_embedding = self.protein_encoder(protein)
        return self.fusion(torch.cat([molecule_embedding, protein_embedding], dim=-1)).squeeze(-1)


def build_features(rows: list[dict[str, object]], n_bits: int, radius: int, protein_dim: int, kmer: int) -> tuple[np.ndarray, np.ndarray]:
    smiles = [str(row["smiles"]) for row in rows]
    sequences = [str(row["protein_sequence"]) for row in rows]

    unique_smiles = sorted(set(smiles))
    unique_sequences = sorted(set(sequences))

    mol_featurizer = MorganFingerprintFeaturizer(
        MorganFingerprintConfig(radius=radius, n_bits=n_bits, use_chirality=True)
    )
    smiles_to_feature = {
        value: feature
        for value, feature in zip(unique_smiles, mol_featurizer.transform(unique_smiles))
    }
    sequence_to_feature = {
        value: feature
        for value, feature in zip(unique_sequences, protein_kmer_feature_matrix(unique_sequences, dim=protein_dim, k=kmer))
    }
    molecule_features = np.stack([smiles_to_feature[value] for value in smiles]).astype(np.float32)
    protein_features = np.stack([sequence_to_feature[value] for value in sequences]).astype(np.float32)
    return molecule_features, protein_features


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            molecule = batch["molecule"].to(device)
            protein = batch["protein"].to(device)
            labels = batch["labels"].to(device)
            logits = model(molecule, protein)
            probs = torch.sigmoid(logits)
            label_values = [int(value) for value in labels.cpu().tolist()]
            prob_values = [float(value) for value in probs.cpu().tolist()]
            y_true.extend(label_values)
            y_score.extend(prob_values)

            for row, label, prob in zip(batch["rows"], label_values, prob_values):
                predictions.append(
                    {
                        "curated_smiles": row["smiles"],
                        "target_chembl_id": row["target_chembl_id"],
                        "uniprot_id": row["uniprot_id"],
                        "molecule_chembl_ids": row["molecule_chembl_ids"],
                        "median_pchembl": row["median_pchembl"],
                        "label": label,
                        "prob_active": prob,
                        "pred_label": int(prob >= 0.5),
                    }
                )

    return binary_classification_metrics(y_true, y_score), predictions


def per_target_metrics(predictions: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["target_chembl_id"])].append(row)

    output: dict[str, dict[str, float]] = {}
    for target, rows in sorted(grouped.items()):
        y_true = [int(row["label"]) for row in rows]
        y_score = [float(row["prob_active"]) for row in rows]
        output[target] = binary_classification_metrics(y_true, y_score)
    return output


def train_one_model(
    model_name: str,
    model: nn.Module,
    loaders: dict[str, DataLoader],
    labels: np.ndarray,
    train_idx: list[int],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, torch.Tensor]]:
    model = model.to(device)
    n_pos = float(labels[train_idx].sum())
    n_neg = float(len(train_idx) - labels[train_idx].sum())
    pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val_auc = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            molecule = batch["molecule"].to(device)
            protein = batch["protein"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(molecule, protein)
            loss = loss_fn(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, loaders["val"], device)
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

    train_metrics, _ = evaluate(model, loaders["train"], device)
    val_metrics, _ = evaluate(model, loaders["val"], device)
    test_metrics, test_predictions = evaluate(model, loaders["test"], device)

    metrics: dict[str, object] = {
        "model": model_name,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "per_target_test": per_target_metrics(test_predictions),
        "history": history,
    }
    return metrics, test_predictions, {key: value.detach().cpu() for key, value in model.state_dict().items()}


def write_report(path: Path, metrics: dict[str, object], split_summary: dict[str, object]) -> None:
    molecule = metrics["models"]["molecule_only_pair"]["test"]
    fusion = metrics["models"]["molecule_protein_kmer_fusion"]["test"]
    delta_auc = fusion["roc_auc"] - molecule["roc_auc"]
    delta_bal = fusion["balanced_accuracy"] - molecule["balanced_accuracy"]
    lines = [
        "# Experiment 3 Comparison Report",
        "",
        "## Question",
        "",
        "Does adding protein/target sequence context improve row-level activity prediction?",
        "",
        "## Split",
        "",
        f"- Split type: `{metrics['split']}`",
        f"- Train rows: `{split_summary['train_rows']}`",
        f"- Validation rows: `{split_summary['val_rows']}`",
        f"- Test rows: `{split_summary['test_rows']}`",
        f"- Train targets: `{', '.join(split_summary['train_targets'])}`",
        f"- Test targets: `{', '.join(split_summary['test_targets'])}`",
        "",
        "## Test Metrics",
        "",
        "| Model | ROC-AUC | PR-AUC | F1 | Balanced Accuracy |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Molecule only | {molecule['roc_auc']:.4f} | {molecule['pr_auc']:.4f} | "
            f"{molecule['f1']:.4f} | {molecule['balanced_accuracy']:.4f} |"
        ),
        (
            f"| Molecule + protein k-mer fusion | {fusion['roc_auc']:.4f} | {fusion['pr_auc']:.4f} | "
            f"{fusion['f1']:.4f} | {fusion['balanced_accuracy']:.4f} |"
        ),
        "",
        "## Insight",
        "",
        f"- ROC-AUC delta: `{delta_auc:+.4f}`",
        f"- Balanced accuracy delta: `{delta_bal:+.4f}`",
        "",
    ]
    if delta_auc > 0.01:
        lines.append("Protein/target sequence context improved ranking performance on this split.")
    elif delta_auc < -0.01:
        lines.append("Protein/target sequence context did not help on this split and may be overfitting or adding noise.")
    else:
        lines.append("Protein/target sequence context gave roughly similar ranking performance on this split.")

    lines.extend(
        [
            "",
            "Interpret this together with scaffold and cold-target splits. A random split can overestimate performance when related molecules appear in both train and test.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 3: molecule-protein fusion insight experiment.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split", choices=["random", "scaffold", "cold_target"], default="scaffold")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--protein-dim", type=int, default=1024)
    parser.add_argument("--protein-kmer", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row limit.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.data)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    molecule_features, protein_features = build_features(
        rows,
        n_bits=args.n_bits,
        radius=args.radius,
        protein_dim=args.protein_dim,
        kmer=args.protein_kmer,
    )
    train_idx, val_idx, test_idx = build_splits(rows, args.split, args.test_size, args.val_size, args.seed)

    loaders = {
        "train": DataLoader(
            PairDataset(rows, molecule_features, protein_features, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        ),
        "val": DataLoader(
            PairDataset(rows, molecule_features, protein_features, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        ),
        "test": DataLoader(
            PairDataset(rows, molecule_features, protein_features, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        ),
    }

    device = select_device(args.device)
    split_summary = {
        "train_rows": len(train_idx),
        "val_rows": len(val_idx),
        "test_rows": len(test_idx),
        "train_targets": sorted({str(rows[index]["target_chembl_id"]) for index in train_idx}),
        "val_targets": sorted({str(rows[index]["target_chembl_id"]) for index in val_idx}),
        "test_targets": sorted({str(rows[index]["target_chembl_id"]) for index in test_idx}),
        "test_label_counts": dict(Counter(int(rows[index]["label"]) for index in test_idx)),
    }

    model_defs = {
        "molecule_only_pair": MoleculeOnlyPairModel(
            molecule_dim=args.n_bits,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
        ),
        "molecule_protein_kmer_fusion": MoleculeProteinFusionModel(
            molecule_dim=args.n_bits,
            protein_dim=args.protein_dim,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
        ),
    }

    all_metrics: dict[str, object] = {
        "experiment": "experiment_3_molecule_protein_fusion",
        "question": "Does adding protein/target sequence context improve molecule-target activity prediction?",
        "input_csv": str(args.data),
        "split": args.split,
        "device": str(device),
        "n_rows": len(rows),
        "n_active": int(labels.sum()),
        "n_inactive": int(len(labels) - labels.sum()),
        "n_unique_smiles": len({str(row["smiles"]) for row in rows}),
        "n_targets": len({str(row["target_chembl_id"]) for row in rows}),
        "split_summary": split_summary,
        "feature": {
            "molecule": {"type": "rdkit_morgan_fingerprint", "n_bits": args.n_bits, "radius": args.radius},
            "protein": {"type": "hashed_amino_acid_kmer", "dim": args.protein_dim, "k": args.protein_kmer},
        },
        "models": {},
        "args": vars(args) | {"data": str(args.data), "out_dir": str(args.out_dir)},
    }

    combined_predictions: list[dict[str, object]] = []
    for model_name, model in model_defs.items():
        metrics, predictions, state = train_one_model(model_name, model, loaders, labels, train_idx, device, args)
        all_metrics["models"][model_name] = metrics
        torch.save(state, args.out_dir / f"{model_name}.pt")
        for prediction in predictions:
            combined = {"model": model_name}
            combined.update(prediction)
            combined_predictions.append(combined)

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))

    predictions_path = args.out_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "curated_smiles",
                "target_chembl_id",
                "uniprot_id",
                "molecule_chembl_ids",
                "median_pchembl",
                "label",
                "prob_active",
                "pred_label",
            ],
        )
        writer.writeheader()
        writer.writerows(combined_predictions)

    report_path = args.out_dir / "comparison_report.md"
    write_report(report_path, all_metrics, split_summary)

    molecule_auc = all_metrics["models"]["molecule_only_pair"]["test"]["roc_auc"]
    fusion_auc = all_metrics["models"]["molecule_protein_kmer_fusion"]["test"]["roc_auc"]
    print("Experiment 3 complete")
    print(f"Split: {args.split}")
    print(f"Rows: {len(rows)}")
    print(f"Device: {device}")
    print(f"Molecule-only test ROC-AUC: {molecule_auc:.4f}")
    print(f"Molecule+protein test ROC-AUC: {fusion_auc:.4f}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
