#!/usr/bin/env python3
"""Experiment 4: course-aligned molecule 3D point-cloud classifier.

3D Vision & Geometry syllabus link:
    Unit III: 3D Point Cloud, Volumetric Representation, Surface/Structure
    representation.

Task:
    SMILES -> RDKit 3D conformer -> atom point cloud -> PointNet-style MLP
    -> active / inactive

This experiment tests whether a learned model can use approximate molecular
3D geometry instead of only 2D fingerprints.

Outputs:
    - metrics.json
    - test_predictions.csv
    - comparison_report.md
    - conformer_failures.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
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

from deep_learning_utils.metrics import binary_classification_metrics  # noqa: E402
from deep_learning_utils.splits import stratified_indices  # noqa: E402


DEFAULT_DATA = REPO_PROJECT_DIR / "multimodal_datapipeline" / "data" / "processed" / "chembl_molecule_curated.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "experiments" / "experiment_4_molecule_3d_pointcloud"


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


def atom_feature_vector(atom: object) -> list[float]:
    atomic_number = float(atom.GetAtomicNum())
    hybridization = str(atom.GetHybridization())
    hybrid_flags = [
        float(hybridization.endswith("SP")),
        float(hybridization.endswith("SP2")),
        float(hybridization.endswith("SP3")),
    ]
    return [
        atomic_number / 100.0,
        float(atom.GetTotalDegree()) / 8.0,
        float(atom.GetFormalCharge()) / 4.0,
        float(atom.GetTotalNumHs()) / 8.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        *hybrid_flags,
    ]


def conformer_to_point_cloud(smiles: str, max_atoms: int, seed: int, optimize_iters: int) -> tuple[np.ndarray, np.ndarray, str | None]:
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required for 3D conformer generation. Use the existing project environment "
            "or install rdkit from conda-forge."
        ) from exc

    RDLogger.DisableLog("rdApp.warning")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((max_atoms, 12), dtype=np.float32), np.zeros((max_atoms,), dtype=np.float32), "invalid_smiles"

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return np.zeros((max_atoms, 12), dtype=np.float32), np.zeros((max_atoms,), dtype=np.float32), "embed_failed"

    if optimize_iters > 0:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=optimize_iters)
        except Exception:
            pass

    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if not heavy_atoms:
        return np.zeros((max_atoms, 12), dtype=np.float32), np.zeros((max_atoms,), dtype=np.float32), "no_heavy_atoms"

    conf = mol.GetConformer()
    rows: list[list[float]] = []
    for atom in heavy_atoms[:max_atoms]:
        pos = conf.GetAtomPosition(atom.GetIdx())
        rows.append([float(pos.x), float(pos.y), float(pos.z), *atom_feature_vector(atom)])

    point_cloud = np.zeros((max_atoms, 12), dtype=np.float32)
    mask = np.zeros((max_atoms,), dtype=np.float32)
    point_cloud[: len(rows)] = np.array(rows, dtype=np.float32)
    mask[: len(rows)] = 1.0

    coords = point_cloud[:, :3]
    valid_coords = coords[mask == 1.0]
    centroid = valid_coords.mean(axis=0, keepdims=True)
    coords -= centroid
    scale = float(np.linalg.norm(coords[mask == 1.0], axis=1).max())
    if scale > 0.0:
        coords /= scale
    point_cloud[:, :3] = coords
    return point_cloud, mask, None


def build_point_clouds(
    examples: list[dict[str, object]],
    max_atoms: int,
    seed: int,
    optimize_iters: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    clouds: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    kept: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for index, example in enumerate(examples):
        cloud, mask, reason = conformer_to_point_cloud(str(example["smiles"]), max_atoms, seed + index, optimize_iters)
        if reason is not None:
            failures.append({"smiles": str(example["smiles"]), "reason": reason})
            continue
        clouds.append(cloud)
        masks.append(mask)
        kept.append(example)

    examples[:] = kept
    if not clouds:
        raise ValueError("No valid 3D conformers were generated.")
    return np.stack(clouds).astype(np.float32), np.stack(masks).astype(np.float32), failures


def random_rotation_matrix() -> torch.Tensor:
    q = torch.randn(4)
    q = q / torch.linalg.norm(q)
    w, x, y, z = q
    return torch.tensor(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


class PointCloudDataset(Dataset):
    def __init__(
        self,
        point_clouds: np.ndarray,
        masks: np.ndarray,
        labels: np.ndarray,
        examples: list[dict[str, object]],
        indices: list[int],
        augment_rotation: bool,
    ) -> None:
        self.point_clouds = point_clouds
        self.masks = masks
        self.labels = labels
        self.examples = examples
        self.indices = indices
        self.augment_rotation = augment_rotation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        index = self.indices[item]
        points = torch.from_numpy(self.point_clouds[index]).float()
        if self.augment_rotation:
            points = points.clone()
            points[:, :3] = points[:, :3] @ random_rotation_matrix().T
        return {
            "index": index,
            "points": points,
            "mask": torch.from_numpy(self.masks[index]).float(),
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
            "example": self.examples[index],
        }


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "indices": [int(item["index"]) for item in batch],
        "points": torch.stack([item["points"] for item in batch]),
        "masks": torch.stack([item["mask"] for item in batch]),
        "labels": torch.stack([item["label"] for item in batch]),
        "examples": [item["example"] for item in batch],
    }


class PointNetClassifier(nn.Module):
    def __init__(self, point_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(point_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, points: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch_size, n_points, point_dim = points.shape
        encoded = self.point_encoder(points.reshape(batch_size * n_points, point_dim))
        encoded = encoded.reshape(batch_size, n_points, -1)
        encoded = encoded.masked_fill(masks.unsqueeze(-1) == 0.0, -1e9)
        pooled = encoded.max(dim=1).values
        return self.classifier(pooled).squeeze(-1)


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
    y_true: list[int] = []
    y_score: list[float] = []
    predictions: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            points = batch["points"].to(device)
            masks = batch["masks"].to(device)
            labels = batch["labels"].to(device)
            logits = model(points, masks)
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


def write_report(path: Path, metrics: dict[str, object]) -> None:
    test = metrics["test"]
    lines = [
        "# Experiment 4 Comparison Report",
        "",
        "## Question",
        "",
        "Can approximate 3D molecular geometry, represented as an atom point cloud, predict molecule activity?",
        "",
        "## Course Syllabus Link",
        "",
        "- Unit III: 3D Point Cloud",
        "- Unit III: Volumetric/structure representation, through normalized xyz coordinates and atom-wise geometric features",
        "- Geometry concept: translation normalization, scale normalization, Euclidean coordinates, and rotation augmentation",
        "",
        "## Model",
        "",
        "```text",
        "SMILES -> RDKit ETKDG 3D conformer -> atom point cloud -> PointNet-style classifier -> active/inactive",
        "```",
        "",
        "## Test Metrics",
        "",
        "| ROC-AUC | PR-AUC | F1 | Balanced Accuracy |",
        "|---:|---:|---:|---:|",
        f"| {test['roc_auc']:.4f} | {test['pr_auc']:.4f} | {test['f1']:.4f} | {test['balanced_accuracy']:.4f} |",
        "",
        "## Interpretation",
        "",
        "This is a direct 3D Vision & Geometry course-topic experiment because the input is a 3D point cloud.",
        "The model is intentionally lightweight; compare it against Experiment 1 to see whether 3D geometry adds useful signal beyond 2D fingerprints.",
        "",
    ]
    path.write_text("\n".join(lines))


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_molecule_examples(args.data, args.activity_threshold)
    if args.max_rows is not None:
        examples = examples[: args.max_rows]

    point_clouds, masks, failures = build_point_clouds(
        examples,
        max_atoms=args.max_atoms,
        seed=args.seed,
        optimize_iters=args.optimize_iters,
    )
    labels = np.array([int(example["label"]) for example in examples], dtype=np.int64)
    if len({int(label) for label in labels}) < 2:
        raise ValueError("Need both active and inactive molecules after conformer generation.")

    train_idx, val_idx, test_idx = stratified_indices(
        labels.tolist(),
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    loaders = {
        "train": DataLoader(
            PointCloudDataset(point_clouds, masks, labels, examples, train_idx, augment_rotation=args.augment_rotation),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_batch,
        ),
        "val": DataLoader(
            PointCloudDataset(point_clouds, masks, labels, examples, val_idx, augment_rotation=False),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        ),
        "test": DataLoader(
            PointCloudDataset(point_clouds, masks, labels, examples, test_idx, augment_rotation=False),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        ),
    }

    device = select_device(args.device)
    model = PointNetClassifier(
        point_dim=point_clouds.shape[-1],
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
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loaders["train"]:
            points = batch["points"].to(device)
            batch_masks = batch["masks"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(points, batch_masks)
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
        "experiment": "experiment_4_molecule_3d_pointcloud",
        "question": "Can approximate 3D molecular geometry predict activity?",
        "course_syllabus_topics": [
            "Unit III: 3D Point Cloud",
            "Unit III: Volumetric Representation",
            "Euclidean geometry: centering, scaling, rotation augmentation",
        ],
        "input_csv": str(args.data),
        "device": str(device),
        "n_rows": len(examples),
        "n_active": int(labels.sum()),
        "n_inactive": int(len(labels) - labels.sum()),
        "n_conformer_failures": len(failures),
        "feature": {
            "type": "rdkit_etkdg_atom_point_cloud",
            "max_atoms": args.max_atoms,
            "point_dim": int(point_clouds.shape[-1]),
            "coordinates": "centered_and_unit_scaled_xyz",
            "atom_features": [
                "atomic_number",
                "degree",
                "formal_charge",
                "hydrogen_count",
                "aromatic",
                "ring",
                "hybridization_flags",
            ],
        },
        "split_summary": {
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "test_rows": len(test_idx),
        },
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "args": vars(args) | {"data": str(args.data), "out_dir": str(args.out_dir)},
    }

    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

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

    failures_path = args.out_dir / "conformer_failures.csv"
    with failures_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["smiles", "reason"])
        writer.writeheader()
        writer.writerows(failures)

    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, args.out_dir / "pointnet_3d.pt")
    write_report(args.out_dir / "comparison_report.md", metrics)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment 4: molecule 3D point-cloud activity classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--activity-threshold", type=float, default=6.0)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--optimize-iters", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max-rows", type=int, default=3000, help="Row cap because 3D conformer generation is slower than fingerprints.")
    parser.add_argument("--augment-rotation", action="store_true", help="Apply random 3D rotations to training point clouds.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metrics = train(args)
    print("Experiment 4 complete")
    print("Course topic: Unit III - 3D Point Cloud")
    print(f"Rows: {metrics['n_rows']}")
    print(f"Conformer failures: {metrics['n_conformer_failures']}")
    print(f"Device: {metrics['device']}")
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Test PR-AUC: {metrics['test']['pr_auc']:.4f}")
    print(f"Wrote outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
