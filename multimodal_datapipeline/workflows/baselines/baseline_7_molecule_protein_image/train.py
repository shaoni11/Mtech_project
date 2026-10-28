#!/usr/bin/env python3
"""Train Baseline 7: molecule + protein + cell-image activity classification.

Rows: aligned (BBBC021 compound, human protein) pairs from ``curate_data.py``.
Image modality: the compound's DMSO-centred mean DINOv2 phenotype profile.

Modalities (--modalities): any subset of molecule, protein, image. Baseline 6
(protein + image) is this trainer with ``--modalities protein,image``; dropping
image gives the same-rows molecule + protein reference.

Integration strategies (proposal Phase 3):
    early fusion  encoders + concat / gated / cross_attention fusion, trained here
    transfer      ``--init-from <baseline_4 model.pt>`` initialises the molecule and
                  protein encoders from a model trained on the large ChEMBL/Davis set
    late fusion   ``--late-fusion-from <baseline_4 model.pt>`` freezes that model and
                  learns an image (+ frozen protein) correction to its logit

Multitask: ``--aux-moa-weight`` adds a MoA classification head on the non-protein
embeddings for rows whose compound has a BBBC021 MoA label.

Because the aligned set is small, ``--seeds`` repeats the split and training and
reports mean and standard deviation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from multimodal_datapipeline.models.fusion import (  # noqa: E402
    ConcatenationFusion,
    CrossAttentionFusion,
    GatedFusion,
    random_modality_mask,
)
from multimodal_datapipeline.models.graph import to_dense_batch  # noqa: E402
from multimodal_datapipeline.models.protein_structure_encoder import (  # noqa: E402
    ProteinGraphConfig,
    ProteinStructureEncoder,
)
from multimodal_datapipeline.utils.baseline_launcher import load_baseline_module  # noqa: E402
from multimodal_datapipeline.utils.metrics import binary_classification_metrics, log_loss, sigmoid  # noqa: E402
from multimodal_datapipeline.utils.microscopy import compound_phenotype_profiles  # noqa: E402
from multimodal_datapipeline.utils.splits import (  # noqa: E402
    group_split,
    scaffold_groups,
    split_overlap,
    stratified_random_split,
)
from multimodal_datapipeline.utils.training import BestEpochTracker, select_device  # noqa: E402


B4 = load_baseline_module("baseline_4_molecule_protein")

DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "baseline_7_molecule_protein_image.csv"
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "interim" / "bbbc021_dinov2_embeddings.npz"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "interim" / "bbbc021_image_metadata.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "baseline_7_molecule_protein_image"
ALL_MODALITIES = ("molecule", "protein", "image")

# Encoder settings that must match a Baseline 4 checkpoint used for transfer or late fusion.
ENCODER_ARGS = (
    "molecule_encoder",
    "protein_encoder",
    "fingerprint_bits",
    "fingerprint_radius",
    "embedding_dim",
    "molecule_hidden_dim",
    "gnn_hidden_dim",
    "molecule_gnn_layers",
    "protein_gnn_layers",
    "contact_cutoff",
    "min_plddt",
    "model_name",
    "max_length",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run baseline_7_molecule_protein_image/curate_data.py first.")
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["label"] in {"0", "1"}]
    if len({row["label"] for row in rows}) < 2:
        raise SystemExit("Need both active and inactive rows.")
    return rows


def phenotype_matrix(rows: list[dict[str, str]], args: argparse.Namespace) -> np.ndarray:
    profiles = compound_phenotype_profiles(args.embeddings, args.metadata, center_on_dmso=not args.no_dmso_centering)
    missing = sorted({row["compound"] for row in rows} - set(profiles))
    if missing:
        raise SystemExit(f"No image embeddings for compounds: {missing[:10]}")
    # Standardise over compounds (label-free, so no label leakage).
    compounds = sorted(profiles)
    stacked = np.stack([profiles[c] for c in compounds])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-6
    return np.stack([(profiles[row["compound"]] - mean) / std for row in rows]).astype(np.float32)


def make_split(rows: list[dict[str, str]], args: argparse.Namespace, seed: int):
    compounds = [row["compound"] for row in rows]
    proteins = [row["uniprot_id"] for row in rows]
    labels = [int(row["label"]) for row in rows]
    if args.split == "compound":
        split = group_split(compounds, args.test_size, args.val_size, seed, balanced=False)
    elif args.split == "scaffold":
        split = group_split(scaffold_groups([row["curated_smiles"] for row in rows]), args.test_size, args.val_size, seed)
    elif args.split == "cold_target":
        split = group_split(proteins, args.test_size, args.val_size, seed, balanced=False)
    else:
        split = stratified_random_split(labels, args.test_size, args.val_size, seed)
    train_idx, val_idx, test_idx = split
    diagnostics = {
        "strategy": args.split,
        "seed": seed,
        "compounds": split_overlap(compounds, train_idx, val_idx, test_idx),
        "proteins": split_overlap(proteins, train_idx, val_idx, test_idx),
        "test_compounds": sorted({compounds[i] for i in test_idx.tolist()}),
    }
    return train_idx, val_idx, test_idx, diagnostics


def load_checkpoint_args(path: Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint


def apply_checkpoint_encoder_args(args: argparse.Namespace, checkpoint_args: dict[str, object]) -> list[str]:
    changed = []
    for key in ENCODER_ARGS:
        if key in checkpoint_args and getattr(args, key, None) != checkpoint_args[key]:
            setattr(args, key, checkpoint_args[key])
            changed.append(key)
    return changed


def transferable_state(source: dict[str, torch.Tensor], target: nn.Module, prefixes: tuple[str, ...]) -> dict:
    target_state = target.state_dict()
    return {
        key: value
        for key, value in source.items()
        if key.startswith(prefixes) and key in target_state and target_state[key].shape == value.shape
    }


class ProteinResources:
    """ESM-2 embeddings / AlphaFold graphs computed once and reused across seeds."""

    def __init__(self, rows: list[dict[str, str]], args: argparse.Namespace, device: torch.device) -> None:
        self.esm = None
        self.graphs = None
        if args.protein_encoder == "esm2":
            sequences = {row["target_chembl_id"]: row["protein_sequence"] for row in rows}
            self.esm = B4.compute_esm_embeddings(sequences, args.model_name, args.max_length, args.protein_batch_size, device)
        else:
            self.graphs = B4.build_protein_graphs(rows, args, device)

    def branch(self, args: argparse.Namespace) -> nn.Module:
        if self.esm is not None:
            return B4.PrecomputedProteinBranch(self.esm, args.embedding_dim)
        encoder = ProteinStructureEncoder(
            graph_config=ProteinGraphConfig(contact_cutoff=args.contact_cutoff, min_plddt=args.min_plddt),
            hidden_dim=args.gnn_hidden_dim,
            embedding_dim=args.embedding_dim,
            num_layers=args.protein_gnn_layers,
            dropout=args.dropout,
        )
        return B4.StructureProteinBranch(self.graphs, encoder)


class MultimodalActivityModel(nn.Module):
    def __init__(
        self,
        modalities: list[str],
        molecule_encoder: nn.Module | None,
        protein_branch: nn.Module | None,
        image_dim: int,
        num_moa: int,
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.modalities = modalities
        self.molecule_encoder = molecule_encoder
        self.protein_branch = protein_branch
        self.fusion_type = args.fusion
        emb = args.embedding_dim
        self.image_projection = (
            nn.Sequential(nn.Dropout(args.dropout), nn.Linear(image_dim, emb), nn.ReLU()) if "image" in modalities else None
        )
        if self.fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                atom_dim=args.gnn_hidden_dim,
                residue_dim=args.gnn_hidden_dim,
                attention_dim=args.gnn_hidden_dim,
                num_heads=args.attention_heads,
                hidden_dim=args.fusion_hidden_dim,
                dropout=args.dropout,
                extra_dims={"image": emb} if "image" in modalities else None,
            )
        elif self.fusion_type == "gated":
            self.fusion = GatedFusion(
                {name: emb for name in modalities},
                fused_dim=emb,
                hidden_dim=args.fusion_hidden_dim,
                dropout=args.dropout,
            )
        else:
            self.fusion = ConcatenationFusion(
                molecule_dim=emb,
                protein_dim=emb,
                image_dim=emb,
                hidden_dim=args.fusion_hidden_dim,
                dropout=args.dropout,
                use_molecule="molecule" in modalities,
                use_protein="protein" in modalities,
                use_image="image" in modalities,
            )
        aux_inputs = [name for name in modalities if name != "protein"]
        self.aux_head = (
            nn.Linear(emb * len(aux_inputs), num_moa) if args.aux_moa_weight > 0 and aux_inputs and num_moa else None
        )

    def embed(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        embeddings = {}
        if self.molecule_encoder is not None:
            embeddings["molecule"] = self.molecule_encoder(batch["molecule"])
        if self.protein_branch is not None:
            embeddings["protein"] = self.protein_branch(batch["protein_keys"])
        if self.image_projection is not None:
            embeddings["image"] = self.image_projection(batch["image"])
        return embeddings

    def forward(
        self, batch: dict[str, object], modality_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        details: dict[str, torch.Tensor] = {}
        embeddings = self.embed(batch) if self.fusion_type != "cross_attention" or self.aux_head is not None else {}
        if self.fusion_type == "cross_attention":
            molecule = batch["molecule"]
            atoms = self.molecule_encoder.encoder.node_embeddings(molecule)
            atom_tokens, atom_mask = to_dense_batch(atoms, molecule.batch, molecule.num_graphs)
            residue_tokens, residue_mask = self.protein_branch.residue_tokens(batch["protein_keys"])
            extra = {"image": self.image_projection(batch["image"])} if self.image_projection is not None else None
            logits, details = self.fusion(
                atom_tokens, atom_mask, residue_tokens, residue_mask, extra=extra, return_attention=True
            )
        elif self.fusion_type == "gated":
            logits, gates = self.fusion(embeddings, return_gates=True, modality_mask=modality_mask)
            details["modality_gates"] = gates
        else:
            logits = self.fusion(
                molecule_embedding=embeddings.get("molecule"),
                protein_embedding=embeddings.get("protein"),
                image_embedding=embeddings.get("image"),
            )
        if self.aux_head is not None:
            details["moa_logits"] = self.aux_head(
                torch.cat([embeddings[name] for name in self.modalities if name != "protein"], dim=-1)
            )
        return logits.squeeze(-1), details


class LateFusionModel(nn.Module):
    """Frozen Baseline 4 logit + learned correction from the image (and frozen protein) embedding."""

    def __init__(self, base_model: nn.Module, image_dim: int, args: argparse.Namespace) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.use_protein = base_model.protein_branch is not None
        input_dim = args.embedding_dim + (args.embedding_dim if self.use_protein else 0)
        self.image_projection = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(image_dim, args.embedding_dim), nn.ReLU())
        self.correction = nn.Sequential(
            nn.Linear(input_dim, args.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fusion_hidden_dim, 1),
        )
        self.base_weight = nn.Parameter(torch.tensor(1.0))
        self.fusion_type = "late"
        self.aux_head = None

    def forward(self, batch: dict[str, object], modality_mask: torch.Tensor | None = None):
        self.base_model.eval()
        with torch.no_grad():
            base_logits = self.base_model(batch["molecule"], batch["protein_keys"])
            protein = self.base_model.protein_branch(batch["protein_keys"]) if self.use_protein else None
        parts = [self.image_projection(batch["image"])] + ([protein] if protein is not None else [])
        correction = self.correction(torch.cat(parts, dim=-1)).squeeze(-1)
        return self.base_weight * base_logits + correction, {"base_logits": base_logits}


def build_model(
    args: argparse.Namespace,
    resources: ProteinResources | None,
    image_dim: int,
    num_moa: int,
    rows: list[dict[str, str]],
    device: torch.device,
) -> nn.Module:
    if args.late_fusion_from:
        checkpoint = load_checkpoint_args(args.late_fusion_from)
        base = B4.MoleculeProteinClassifier(
            molecule_encoder=B4.build_molecule_encoder(args),
            protein_branch=resources.branch(args) if resources else None,
            fusion=checkpoint["args"].get("fusion", "concat"),
            embedding_dim=args.embedding_dim,
            gnn_hidden_dim=args.gnn_hidden_dim,
            fusion_hidden_dim=checkpoint["args"].get("fusion_hidden_dim", args.fusion_hidden_dim),
            dropout=args.dropout,
            attention_heads=checkpoint["args"].get("attention_heads", args.attention_heads),
        )
        state = {k: v for k, v in checkpoint["model_state_dict"].items() if not k.endswith(".table")}
        base.load_state_dict(state, strict=False)
        return LateFusionModel(base, image_dim, args).to(device)

    model = MultimodalActivityModel(
        modalities=args.modalities,
        molecule_encoder=B4.build_molecule_encoder(args) if "molecule" in args.modalities else None,
        protein_branch=resources.branch(args) if "protein" in args.modalities else None,
        image_dim=image_dim,
        num_moa=num_moa,
        args=args,
    )
    if args.init_from:
        checkpoint = load_checkpoint_args(args.init_from)
        state = transferable_state(
            checkpoint["model_state_dict"], model, ("molecule_encoder.", "protein_branch.encoder.", "protein_branch.projection.")
        )
        model.load_state_dict(state, strict=False)
        print(f"  transferred {len(state)} tensors from {args.init_from}", flush=True)
    return model.to(device)


def make_loader(rows, molecule_inputs, indices, args, shuffle):
    dataset = B4.MoleculeProteinDataset(rows, molecule_inputs, indices)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=B4.collate_batch)


def to_device(batch, images: torch.Tensor, moa_ids: torch.Tensor, device: torch.device) -> dict[str, object]:
    row_idx = batch["row_idx"]
    return {
        "molecule": batch["molecule"].to(device),
        "protein_keys": batch["protein_keys"],
        "image": images[row_idx].to(device),
        "moa": moa_ids[row_idx].to(device),
        "label": batch["label"].to(device),
        "row_idx": row_idx,
    }


def evaluate(model, loader, images, moa_ids, device):
    model.eval()
    y_true, logits_out, row_indices, gates = [], [], [], []
    with torch.no_grad():
        for raw in loader:
            batch = to_device(raw, images, moa_ids, device)
            logits, details = model(batch)
            y_true.extend(batch["label"].cpu().tolist())
            logits_out.extend(logits.cpu().tolist())
            row_indices.extend(batch["row_idx"].tolist())
            if "modality_gates" in details:
                gates.extend(details["modality_gates"].cpu().tolist())
    y = np.array(y_true, dtype=np.int32)
    probs = sigmoid(np.array(logits_out, dtype=np.float32))
    metrics = {"loss": log_loss(y.astype(np.float32), probs), **binary_classification_metrics(y, probs), "n_examples": len(y)}
    if gates:
        mean_gates = np.mean(np.array(gates), axis=0)
        metrics.update({f"mean_gate_{name}": float(mean_gates[i]) for i, name in enumerate(model.modalities)})
    return metrics, list(zip(row_indices, y.tolist(), probs.tolist()))


def run_seed(seed, rows, molecule_inputs, images, moa_ids, num_moa, resources, args, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_idx, val_idx, test_idx, diagnostics = make_split(rows, args, seed)
    model = build_model(args, resources, images.shape[1], num_moa, rows, device)
    loaders = {
        "train": make_loader(rows, molecule_inputs, train_idx, args, True),
        "val": make_loader(rows, molecule_inputs, val_idx, args, False),
        "test": make_loader(rows, molecule_inputs, test_idx, args, False),
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay
    )
    tracker = BestEpochTracker(patience=args.patience)
    n_modalities = len(getattr(model, "modalities", []))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for raw in loaders["train"]:
            batch = to_device(raw, images, moa_ids, device)
            mask = None
            if args.modality_dropout > 0 and model.fusion_type == "gated" and n_modalities > 1:
                mask = random_modality_mask(len(batch["label"]), n_modalities, args.modality_dropout, device)
            logits, details = model(batch, mask)
            loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
            if "moa_logits" in details and (batch["moa"] >= 0).any():
                has_moa = batch["moa"] >= 0
                loss = loss + args.aux_moa_weight * F.cross_entropy(details["moa_logits"][has_moa], batch["moa"][has_moa])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _ = evaluate(model, loaders["val"], images, moa_ids, device)
        tracker.update(epoch, val_metrics["roc_auc"], model)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_roc_auc": val_metrics["roc_auc"]})
        if tracker.should_stop:
            break
    tracker.restore(model)
    test_metrics, test_predictions = evaluate(model, loaders["test"], images, moa_ids, device)
    val_metrics, _ = evaluate(model, loaders["val"], images, moa_ids, device)
    return {
        "seed": seed,
        "split": diagnostics,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "model_selection": tracker.summary() | {"epochs_run": len(history)},
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
    }, test_predictions, model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Baseline 7 molecule + protein + image model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--modalities", default="molecule,protein,image", help="Comma list of molecule, protein, image.")
    parser.add_argument("--split", default="compound", choices=["compound", "scaffold", "cold_target", "random"])
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma list of split/training seeds.")
    parser.add_argument("--molecule-encoder", default="fingerprint", choices=["fingerprint", "gnn"])
    parser.add_argument("--protein-encoder", default="esm2", choices=["esm2", "structure_gnn"])
    parser.add_argument("--fusion", default="gated", choices=["concat", "gated", "cross_attention"])
    parser.add_argument("--init-from", type=Path, default=None, help="Baseline 4 model.pt for encoder transfer.")
    parser.add_argument("--late-fusion-from", type=Path, default=None, help="Baseline 4 model.pt to freeze (late fusion).")
    parser.add_argument("--aux-moa-weight", type=float, default=0.0, help="Weight of the auxiliary MoA loss (multitask).")
    parser.add_argument("--modality-dropout", type=float, default=0.0)
    parser.add_argument("--no-dmso-centering", action="store_true")
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--protein-batch-size", type=int, default=2)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--molecule-hidden-dim", type=int, default=512)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--molecule-gnn-layers", type=int, default=4)
    parser.add_argument("--protein-gnn-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--min-plddt", type=float, default=0.0)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    if not args.modalities or set(args.modalities) - set(ALL_MODALITIES):
        raise SystemExit(f"--modalities must be a non-empty subset of {ALL_MODALITIES}")
    args.modalities = [m for m in ALL_MODALITIES if m in args.modalities]
    seeds = [int(seed) for seed in args.seeds.split(",")]
    checkpoint_path = args.late_fusion_from or args.init_from
    if checkpoint_path:
        changed = apply_checkpoint_encoder_args(args, load_checkpoint_args(checkpoint_path)["args"])
        if changed:
            print(f"Using encoder settings from {checkpoint_path}: {changed}", flush=True)
    if args.late_fusion_from:
        args.modalities = ["molecule", "image"] + (["protein"] if args.protein_encoder != "none" else [])
    if args.fusion == "cross_attention" and (
        args.molecule_encoder != "gnn" or args.protein_encoder != "structure_gnn"
        or not {"molecule", "protein"} <= set(args.modalities)
    ):
        raise SystemExit("--fusion cross_attention needs molecule+protein with --molecule-encoder gnn --protein-encoder structure_gnn")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.data)
    device = select_device(args.device)
    images = torch.from_numpy(phenotype_matrix(rows, args))
    moa_names = sorted({row["moa"] for row in rows if row["moa"]})
    moa_ids = torch.tensor([moa_names.index(row["moa"]) if row["moa"] else -1 for row in rows], dtype=torch.long)
    needs_molecule = "molecule" in args.modalities
    molecule_inputs = B4.build_molecule_inputs(rows, args) if needs_molecule else [torch.zeros(1)] * len(rows)
    resources = ProteinResources(rows, args, device) if "protein" in args.modalities and args.protein_encoder != "none" else None
    print(
        f"Rows: {len(rows)}  compounds: {len({r['compound'] for r in rows})}  proteins: {len({r['uniprot_id'] for r in rows})}  "
        f"modalities: {args.modalities}  fusion: {'late' if args.late_fusion_from else args.fusion}  split: {args.split}",
        flush=True,
    )

    runs, all_predictions = [], []
    for seed in seeds:
        result, predictions, model = run_seed(seed, rows, molecule_inputs, images, moa_ids, len(moa_names), resources, args, device)
        runs.append(result)
        all_predictions.extend({"seed": seed, "row_idx": r, "label": y, "probability": p} for r, y, p in predictions)
        print(
            f"seed {seed}: test ROC-AUC={result['test']['roc_auc']:.4f} PR-AUC={result['test']['pr_auc']:.4f} "
            f"(best epoch {result['model_selection']['best_epoch']}, test compounds {len(result['split']['test_compounds'])})",
            flush=True,
        )
        if seed == seeds[-1]:
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args) | {"seed": seed}}, args.out_dir / "model.pt")

    summary = {}
    for metric in ("roc_auc", "pr_auc", "balanced_accuracy", "f1", "accuracy"):
        values = np.array([run["test"][metric] for run in runs], dtype=np.float64)
        values = values[~np.isnan(values)]
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std()) if len(values) else float("nan"),
            "n": int(len(values)),
        }
    gate_keys = [key for key in runs[0]["test"] if key.startswith("mean_gate_")]
    for key in gate_keys:
        summary[key] = float(np.mean([run["test"][key] for run in runs]))

    metrics = {
        "task": "molecule_protein_image_activity_classification",
        "modalities": args.modalities,
        "strategy": "late_fusion" if args.late_fusion_from else ("transfer" if args.init_from else "early_fusion"),
        "fusion": "late" if args.late_fusion_from else args.fusion,
        "n_rows": len(rows),
        "test_summary": summary,
        "runs": runs,
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
    }
    with (args.out_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (args.out_dir / "test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "row_idx", "compound", "uniprot_id", "label", "probability"])
        writer.writeheader()
        for prediction in all_predictions:
            row = rows[prediction["row_idx"]]
            writer.writerow(prediction | {"compound": row["compound"], "uniprot_id": row["uniprot_id"]})

    print("Baseline 7 complete")
    print(f"Test ROC-AUC {summary['roc_auc']['mean']:.4f} ± {summary['roc_auc']['std']:.4f} over {summary['roc_auc']['n']} seeds")
    print(f"Test PR-AUC  {summary['pr_auc']['mean']:.4f} ± {summary['pr_auc']['std']:.4f}")
    for key in gate_keys:
        print(f"{key}: {summary[key]:.3f}")
    print(f"Wrote metrics: {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
