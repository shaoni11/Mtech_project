#!/usr/bin/env python3
"""Train Baseline 5: molecule + cell-image MoA classification on BBBC021.

Unit of prediction: a treatment (compound, concentration). Its phenotype is the
plate-wise DMSO-centred mean DINOv2 embedding of its images (from
``baseline_3_image_only/extract_image_embeddings.py``); its molecule input is the
compound's Morgan fingerprint.

Modalities (--modalities): any of molecule, image. With both, a GatedFusion head
reports how much weight each modality receives; ``--modality-dropout`` randomly
hides a modality during training so one modality cannot dominate. All modality
combinations use the same treatments (compounds with SMILES) unless
``--include-compounds-without-smiles`` is given for an image-only run.

Evaluation (--split):
    loco      leave-one-compound-out cross-validation (standard BBBC021 NSC protocol)
    compound  one held-out-compound split stratified by MoA

``--model nn1`` is the classic 1-nearest-neighbour (cosine) image baseline with no
training, for comparison with published BBBC021 NSC accuracies.
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from multimodal_datapipeline.models.fusion import GatedFusion, random_modality_mask  # noqa: E402
from multimodal_datapipeline.models.molecule_encoder import (  # noqa: E402
    MorganFingerprintConfig,
    MorganFingerprintFeaturizer,
)
from multimodal_datapipeline.utils.microscopy import (  # noqa: E402
    aggregate_profiles,
    dmso_center,
    read_embedding_table,
)
from multimodal_datapipeline.utils.splits import stratified_group_split  # noqa: E402
from multimodal_datapipeline.utils.training import select_device  # noqa: E402


DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "interim" / "bbbc021_dinov2_embeddings.npz"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "interim" / "bbbc021_image_metadata.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "baseline_5_molecule_image"


def build_treatments(args: argparse.Namespace) -> list[dict[str, object]]:
    embeddings, metadata = read_embedding_table(args.embeddings, args.metadata)
    if not args.no_dmso_centering:
        embeddings = dmso_center(embeddings, metadata)
    profiles = aggregate_profiles(embeddings, metadata, ("compound", "concentration"))

    info: dict[tuple[str, str], dict[str, str]] = {}
    image_counts: Counter = Counter()
    for row in metadata:
        key = (row["compound"], row["concentration"])
        info.setdefault(key, row)
        image_counts[key] += 1

    treatments = []
    for key, profile in sorted(profiles.items()):
        row = info[key]
        if not row["moa"] or row["moa"] == "DMSO":
            continue
        if not row["smiles"] and not args.include_compounds_without_smiles:
            continue  # keep the same treatments for every modality combination
        treatments.append(
            {
                "compound": key[0],
                "concentration": key[1],
                "moa": row["moa"],
                "smiles": row["smiles"],
                "n_images": image_counts[key],
                "profile": profile.astype(np.float32),
            }
        )
    return treatments


def fingerprint_features(treatments: list[dict[str, object]], n_bits: int) -> np.ndarray:
    featurizer = MorganFingerprintFeaturizer(MorganFingerprintConfig(n_bits=n_bits))
    by_smiles = {}
    for treatment in treatments:
        smiles = str(treatment["smiles"])
        if smiles and smiles not in by_smiles:
            by_smiles[smiles] = featurizer.transform_one(smiles)
    return np.stack(
        [by_smiles.get(str(t["smiles"]), np.zeros(n_bits, dtype=np.float32)) for t in treatments]
    ).astype(np.float32)


def make_folds(
    treatments: list[dict[str, object]], args: argparse.Namespace
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    compounds = [str(t["compound"]) for t in treatments]
    if args.split == "loco":
        folds = []
        for compound in sorted(set(compounds)):
            test = np.array([i for i, c in enumerate(compounds) if c == compound])
            train = np.array([i for i, c in enumerate(compounds) if c != compound])
            folds.append((compound, train, test))
        return folds
    (train, val, test), _ = stratified_group_split(
        compounds, [str(t["moa"]) for t in treatments], args.test_size, 0.0, args.seed
    )
    return [("compound_split", np.concatenate([train, val]), test)]


class MoaClassifier(nn.Module):
    def __init__(self, input_dims: dict[str, int], num_classes: int, args: argparse.Namespace) -> None:
        super().__init__()
        self.fusion = GatedFusion(
            input_dims,
            fused_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=num_classes,
            dropout=args.dropout,
        )

    def forward(
        self, inputs: dict[str, torch.Tensor], modality_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fusion(inputs, return_gates=True, modality_mask=modality_mask)


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mean) / std, (other - mean) / std


def train_fold(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    inputs_train, inputs_test = {}, {}
    for name, matrix in features.items():
        if name == "image":
            inputs_train[name], inputs_test[name] = standardize(matrix[train_idx], matrix[test_idx])
        else:
            inputs_train[name], inputs_test[name] = matrix[train_idx], matrix[test_idx]

    torch.manual_seed(args.seed)
    model = MoaClassifier({name: m.shape[1] for name, m in features.items()}, num_classes, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    counts = np.bincount(labels[train_idx], minlength=num_classes)
    weights = torch.tensor(
        [len(train_idx) / max(1, num_classes * count) if count else 0.0 for count in counts], dtype=torch.float32
    ).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    x_train = {name: torch.from_numpy(m).to(device) for name, m in inputs_train.items()}
    y_train = torch.from_numpy(labels[train_idx]).long().to(device)

    for _ in range(args.epochs):
        model.train()
        mask = None
        if args.modality_dropout > 0 and len(features) > 1:
            mask = random_modality_mask(len(train_idx), len(features), args.modality_dropout, device)
        logits, _ = model(x_train, mask)
        loss = loss_fn(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, gates = model({name: torch.from_numpy(m).to(device) for name, m in inputs_test.items()})
    return torch.softmax(logits, dim=-1).cpu().numpy(), gates.cpu().numpy()


def nearest_neighbour(
    profiles: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, num_classes: int
) -> np.ndarray:
    train = profiles[train_idx] / np.linalg.norm(profiles[train_idx], axis=1, keepdims=True).clip(1e-8)
    test = profiles[test_idx] / np.linalg.norm(profiles[test_idx], axis=1, keepdims=True).clip(1e-8)
    nearest = np.argmax(test @ train.T, axis=1)
    probs = np.zeros((len(test_idx), num_classes), dtype=np.float32)
    probs[np.arange(len(test_idx)), labels[train_idx][nearest]] = 1.0
    return probs


def macro_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    classes = sorted(set(y_true) | set(y_pred))
    f1s, recalls = [], []
    for class_id in classes:
        tp = sum(t == class_id and p == class_id for t, p in zip(y_true, y_pred))
        fp = sum(t != class_id and p == class_id for t, p in zip(y_true, y_pred))
        fn = sum(t == class_id and p != class_id for t, p in zip(y_true, y_pred))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1s.append(2 * precision * recall / max(1e-12, precision + recall))
        recalls.append(recall)
    return {
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / max(1, len(y_true)),
        "macro_f1": float(np.mean(f1s)),
        "macro_recall": float(np.mean(recalls)),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 5 molecule + image MoA classifier.")
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--modalities", default="molecule,image", help="Comma list of: molecule, image.")
    parser.add_argument("--model", default="gated", choices=["gated", "nn1"])
    parser.add_argument("--split", default="loco", choices=["loco", "compound"])
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=0.0,
        help="Probability of hiding each modality per training example (modality masking).",
    )
    parser.add_argument("--no-dmso-centering", action="store_true")
    parser.add_argument(
        "--include-compounds-without-smiles",
        action="store_true",
        help="Image-only literature comparison: also use the AZ compounds that have no SMILES.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    unknown = set(args.modalities) - {"molecule", "image"}
    if unknown or not args.modalities:
        raise SystemExit(f"--modalities must be a non-empty subset of molecule,image (got {args.modalities})")
    if args.include_compounds_without_smiles and "molecule" in args.modalities:
        raise SystemExit("--include-compounds-without-smiles only applies to --modalities image")
    if args.model == "nn1" and args.modalities != ["image"]:
        raise SystemExit("--model nn1 is the image-only nearest-neighbour baseline: use --modalities image")
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    treatments = build_treatments(args)
    moa_labels = sorted({str(t["moa"]) for t in treatments})
    label_to_id = {label: i for i, label in enumerate(moa_labels)}
    labels = np.array([label_to_id[str(t["moa"])] for t in treatments], dtype=np.int64)
    profiles = np.stack([t["profile"] for t in treatments])
    features = {}
    if "molecule" in args.modalities:
        features["molecule"] = fingerprint_features(treatments, args.fingerprint_bits)
    if "image" in args.modalities:
        features["image"] = profiles

    print(
        f"Treatments: {len(treatments)}  compounds: {len({t['compound'] for t in treatments})}  "
        f"MoA classes: {len(moa_labels)}  modalities: {args.modalities}  model: {args.model}",
        flush=True,
    )
    device = select_device(args.device)
    folds = make_folds(treatments, args)

    predictions = []
    gate_rows = []
    for fold_name, train_idx, test_idx in folds:
        if args.model == "nn1":
            probs = nearest_neighbour(profiles, labels, train_idx, test_idx, len(moa_labels))
            gates = None
        else:
            probs, gates = train_fold(features, labels, train_idx, test_idx, len(moa_labels), args, device)
        for position, row_idx in enumerate(test_idx.tolist()):
            treatment = treatments[row_idx]
            predictions.append(
                {
                    "fold": fold_name,
                    "compound": treatment["compound"],
                    "concentration": treatment["concentration"],
                    "true_moa": treatment["moa"],
                    "predicted_moa": moa_labels[int(np.argmax(probs[position]))],
                    "confidence": float(np.max(probs[position])),
                    **(
                        {f"gate_{name}": float(gates[position][i]) for i, name in enumerate(features)}
                        if gates is not None
                        else {}
                    ),
                }
            )
        if gates is not None:
            gate_rows.extend(gates.tolist())

    y_true = [label_to_id[p["true_moa"]] for p in predictions]
    y_pred = [label_to_id[p["predicted_moa"]] for p in predictions]
    treatment_metrics = macro_metrics(y_true, y_pred)

    # Compound-level vote (majority over a compound's concentrations), as often reported for BBBC021.
    by_compound: dict[str, list[dict[str, object]]] = defaultdict(list)
    for prediction in predictions:
        by_compound[str(prediction["compound"])].append(prediction)
    compound_true, compound_pred = [], []
    for compound_predictions in by_compound.values():
        compound_true.append(label_to_id[str(compound_predictions[0]["true_moa"])])
        votes = Counter(str(p["predicted_moa"]) for p in compound_predictions)
        compound_pred.append(label_to_id[votes.most_common(1)[0][0]])
    compound_metrics = macro_metrics(compound_true, compound_pred)

    metrics = {
        "task": "molecule_image_moa_classification",
        "unit": "treatment (compound, concentration) DMSO-centred mean DINOv2 profile",
        "modalities": args.modalities,
        "model": args.model,
        "split": args.split,
        "n_folds": len(folds),
        "n_treatments": len(treatments),
        "n_compounds": len(by_compound),
        "classes": moa_labels,
        "treatment_level": treatment_metrics,
        "compound_level": compound_metrics,
        "mean_gates": (
            {name: float(np.mean([g[i] for g in gate_rows])) for i, name in enumerate(features)} if gate_rows else None
        ),
        "args": vars(args) | {key: str(vars(args)[key]) for key in ("embeddings", "metadata", "out_dir")},
    }
    with (args.out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.out_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    print("Molecule + image baseline complete")
    print(f"Treatment-level accuracy (NSC-style): {treatment_metrics['accuracy']:.4f}  "
          f"macro-F1: {treatment_metrics['macro_f1']:.4f}")
    print(f"Compound-level accuracy: {compound_metrics['accuracy']:.4f}")
    if metrics["mean_gates"]:
        print(f"Mean modality gates: {metrics['mean_gates']}")
    print(f"Wrote metrics: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
