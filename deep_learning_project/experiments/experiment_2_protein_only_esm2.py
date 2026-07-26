#!/usr/bin/env python3
"""Experiment 2: protein-only sanity baseline.
goal: Can protein sequence alone predict target-level active_fraction?

Task:
    protein sequence -> frozen ESM-2 embedding -> MLP -> active_fraction

This is intentionally a sanity experiment. The available processed table has
only 12 target-level rows, so the output should not be treated as a strong
standalone deep-learning result.

Interpretation: This is only a sanity check. The dataset has just 12 protein-level rows, so the metric is unstable and not thesis-strong by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_PROJECT_DIR = PROJECT_DIR.parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from deep_learning_utils.metrics import regression_metrics  # noqa: E402
from deep_learning_utils.splits import random_row_splits  # noqa: E402


DEFAULT_DATA = REPO_PROJECT_DIR / "multimodal_datapipeline" / "data" / "processed" / "baseline_2_protein_only.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "experiments" / "experiment_2_protein_only_esm2"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "target_chembl_id",
        "uniprot_id",
        "pref_name",
        "protein_sequence",
        "sequence_length",
        "active_fraction",
        "n_molecule_target_rows",
    }
    missing = required.difference(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")

    usable = [
        row
        for row in rows
        if row["protein_sequence"].strip() and row["active_fraction"].strip()
    ]
    if len(usable) < 4:
        raise ValueError(f"Need at least 4 usable protein rows, found {len(usable)}")
    return usable


class ProteinDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], indices: list[int]) -> None:
        self.rows = rows
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.rows[self.indices[item]]
        return {
            "sequence": row["protein_sequence"],
            "target": torch.tensor(float(row["active_fraction"]), dtype=torch.float32),
            "target_chembl_id": row["target_chembl_id"],
            "uniprot_id": row["uniprot_id"],
            "pref_name": row["pref_name"],
            "sequence_length": row["sequence_length"],
            "n_molecule_target_rows": row["n_molecule_target_rows"],
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sequences": [str(item["sequence"]) for item in batch],
        "targets": torch.stack([item["target"] for item in batch]),
        "target_chembl_ids": [str(item["target_chembl_id"]) for item in batch],
        "uniprot_ids": [str(item["uniprot_id"]) for item in batch],
        "pref_names": [str(item["pref_name"]) for item in batch],
        "sequence_lengths": [str(item["sequence_length"]) for item in batch],
        "n_molecule_target_rows": [str(item["n_molecule_target_rows"]) for item in batch],
    }


class ProteinOnlyRegressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        max_length: int,
        fine_tune_backbone: bool,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Install transformers before using ESM-2: pip install transformers"
            ) from exc

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = nn.Linear(hidden_size, embedding_dim)
        if not fine_tune_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, sequences: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)
        outputs = self.backbone(**tokens)
        residue_embeddings = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(residue_embeddings.dtype)
        if mask.shape[1] > 2:
            residue_embeddings = residue_embeddings[:, 1:-1, :]
            mask = mask[:, 1:-1, :]
        embeddings = (residue_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        embeddings = self.projection(embeddings)
        return self.head(embeddings).squeeze(-1)


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[float] = []
    y_pred: list[float] = []
    prediction_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].to(device)
            outputs = torch.sigmoid(model(batch["sequences"]))
            true_values = [float(value) for value in targets.cpu().tolist()]
            pred_values = [float(value) for value in outputs.cpu().tolist()]
            y_true.extend(true_values)
            y_pred.extend(pred_values)

            for target_id, uniprot_id, pref_name, seq_len, n_rows, true, pred in zip(
                batch["target_chembl_ids"],
                batch["uniprot_ids"],
                batch["pref_names"],
                batch["sequence_lengths"],
                batch["n_molecule_target_rows"],
                true_values,
                pred_values,
            ):
                prediction_rows.append(
                    {
                        "target_chembl_id": target_id,
                        "uniprot_id": uniprot_id,
                        "pref_name": pref_name,
                        "sequence_length": seq_len,
                        "n_molecule_target_rows": n_rows,
                        "active_fraction": true,
                        "predicted_active_fraction": pred,
                    }
                )

    return regression_metrics(y_true, y_pred), prediction_rows


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.data)
    train_idx, val_idx, test_idx = random_row_splits(
        len(rows),
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    train_loader = DataLoader(
        ProteinDataset(rows, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        ProteinDataset(rows, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        ProteinDataset(rows, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = select_device(args.device)
    try:
        model = ProteinOnlyRegressor(
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            max_length=args.max_length,
            fine_tune_backbone=args.fine_tune_backbone,
        ).to(device)
    except OSError as exc:
        raise SystemExit(
            "Could not load the ESM-2 pretrained model. The first run needs access "
            "to Hugging Face to download the model, or pass --model-name pointing "
            "to a local model directory.\n"
            f"Original error: {exc}"
        ) from exc

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            targets = batch["targets"].to(device).clamp(0.0, 1.0)
            predictions = torch.sigmoid(model(batch["sequences"]))
            loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, val_loader, device)
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": sum(losses) / max(1, len(losses)),
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
            }
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics, _ = evaluate(model, train_loader, device)
    val_metrics, _ = evaluate(model, val_loader, device)
    test_metrics, test_predictions = evaluate(model, test_loader, device)

    metrics: dict[str, object] = {
        "experiment": "experiment_2_protein_only_esm2",
        "task": "protein_sequence_to_target_active_fraction",
        "input_csv": str(args.data),
        "model": {
            "protein_encoder": args.model_name,
            "freeze_backbone": not args.fine_tune_backbone,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "max_length": args.max_length,
        },
        "target_column": "active_fraction",
        "device": str(device),
        "n_rows": len(rows),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "caveat": (
            "Sanity baseline only: the available protein-only table has 12 target-level rows, "
            "so validation/test metrics are high variance and not suitable as a standalone claim."
        ),
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
                "target_chembl_id",
                "uniprot_id",
                "pref_name",
                "sequence_length",
                "n_molecule_target_rows",
                "active_fraction",
                "predicted_active_fraction",
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
    parser = argparse.ArgumentParser(description="Experiment 2: protein-only ESM-2 sanity baseline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--fine-tune-backbone", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = train(args)
    metrics = result["metrics"]
    print("Experiment 2 complete")
    print(f"Rows: {metrics['n_rows']}")
    print(f"Device: {metrics['device']}")
    print(f"Test metrics: {metrics['test']}")
    print(f"Wrote metrics: {result['metrics_path']}")
    print(f"Wrote predictions: {result['predictions_path']}")


if __name__ == "__main__":
    main()
