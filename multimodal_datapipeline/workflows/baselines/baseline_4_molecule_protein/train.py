#!/usr/bin/env python3
"""Train Baseline 4: molecule + protein activity classification.

Molecule branch (--molecule-encoder):
    fingerprint  curated SMILES -> Morgan fingerprint -> MLP
    gnn          curated SMILES -> atom/bond graph -> message-passing GNN

Protein branch (--protein-encoder):
    esm2           sequence -> frozen ESM-2 mean embedding (cached) -> trainable projection
    structure_gnn  AlphaFold PDB -> residue contact graph -> trainable GNN
    none           no protein input: same rows, split and molecule branch (modality ablation)

Fusion (--fusion):
    concat           concatenation MLP
    gated            learned per-example modality gates (gates are reported)
    cross_attention  molecule atoms attend over protein residues (needs gnn + structure_gnn)

Splits (--split):
    random       stratified random rows (optimistic)
    scaffold     Bemis-Murcko scaffold groups; no test scaffold is seen in training
    cold_target  whole proteins (UniProt) held out; mutants of a held-out gene are held out too

Data: the ChEMBL kinase table (default) or the Davis benchmark
(``--data data/processed/baseline_4_davis.csv`` from ``curate_davis.py``). Each
distinct ``target_chembl_id`` is one protein construct, so Davis mutants get their
own sequence (ESM-2) or mutated residue identities (structure GNN).

The epoch with the best validation ROC-AUC is restored before testing
(``--patience`` adds early stopping).
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

from multimodal_datapipeline.models.fusion import (  # noqa: E402
    CrossAttentionFusion,
    GatedFusion,
    TwoModalityFusion,
)
from multimodal_datapipeline.models.graph import Graph, GraphBatch, batch_graphs, to_dense_batch  # noqa: E402
from multimodal_datapipeline.models.molecule_encoder import (  # noqa: E402
    MoleculeFingerprintMLP,
    MoleculeGNNEncoder,
    MoleculeGraphFeaturizer,
    MorganFingerprintConfig,
    MorganFingerprintFeaturizer,
)
from multimodal_datapipeline.models.protein_encoder import ESM2ProteinEncoder  # noqa: E402
from multimodal_datapipeline.models.protein_structure_encoder import (  # noqa: E402
    ProteinGraphConfig,
    ProteinStructureEncoder,
    ProteinStructureGraphBuilder,
)
from multimodal_datapipeline.utils.metrics import (  # noqa: E402
    binary_classification_metrics,
    log_loss,
    sigmoid,
)
from multimodal_datapipeline.utils.splits import (  # noqa: E402
    group_split,
    scaffold_groups,
    split_overlap,
    stratified_random_split,
)
from multimodal_datapipeline.utils.training import BestEpochTracker, select_device  # noqa: E402


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
        "alphafold_pdb_path",
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


def make_split(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    labels = [int(row["label"]) for row in rows]
    smiles = [row["curated_smiles"] for row in rows]
    targets = [row["target_chembl_id"] for row in rows]
    proteins = [row["uniprot_id"] for row in rows]
    scaffolds = scaffold_groups(smiles)

    if args.split == "random":
        train_idx, val_idx, test_idx = stratified_random_split(labels, args.test_size, args.val_size, args.seed)
    elif args.split == "scaffold":
        train_idx, val_idx, test_idx = group_split(scaffolds, args.test_size, args.val_size, args.seed)
    elif args.split == "cold_target":
        train_idx, val_idx, test_idx = group_split(
            proteins, args.test_size, args.val_size, args.seed, balanced=False
        )
    else:
        raise ValueError(f"Unknown split: {args.split}")

    diagnostics = {
        "strategy": args.split,
        "molecules": split_overlap(smiles, train_idx, val_idx, test_idx),
        "scaffolds": split_overlap(scaffolds, train_idx, val_idx, test_idx),
        "targets": split_overlap(targets, train_idx, val_idx, test_idx),
        "uniprot_ids": split_overlap(proteins, train_idx, val_idx, test_idx),
        "test_targets": sorted({targets[i] for i in test_idx.tolist()}),
        "val_targets": sorted({targets[i] for i in val_idx.tolist()}),
    }
    return train_idx, val_idx, test_idx, diagnostics


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate if candidate.exists() else path


def build_molecule_inputs(rows: list[dict[str, str]], args: argparse.Namespace) -> list[object]:
    """One model input per row: a fingerprint tensor or a molecular Graph."""
    unique_smiles = sorted({row["curated_smiles"] for row in rows})
    if args.molecule_encoder == "fingerprint":
        featurizer = MorganFingerprintFeaturizer(
            MorganFingerprintConfig(radius=args.fingerprint_radius, n_bits=args.fingerprint_bits)
        )
        matrix = featurizer.transform(unique_smiles)
        by_smiles = {smiles: torch.from_numpy(matrix[i]) for i, smiles in enumerate(unique_smiles)}
    else:
        featurizer = MoleculeGraphFeaturizer()
        by_smiles = {smiles: featurizer.transform_one(smiles) for smiles in unique_smiles}
    return [by_smiles[row["curated_smiles"]] for row in rows]


class PrecomputedProteinBranch(nn.Module):
    """Frozen ESM-2 embeddings looked up by protein key, then a trainable projection."""

    def __init__(self, embeddings: dict[str, torch.Tensor], embedding_dim: int) -> None:
        super().__init__()
        self.index = {key: i for i, key in enumerate(sorted(embeddings))}
        table = torch.stack([embeddings[key] for key in sorted(embeddings)])
        self.register_buffer("table", table)
        self.projection = nn.Sequential(nn.Linear(table.shape[1], embedding_dim), nn.ReLU())

    def forward(self, protein_keys: list[str]) -> torch.Tensor:
        rows = torch.tensor([self.index[key] for key in protein_keys], device=self.table.device)
        return self.projection(self.table[rows])


class StructureProteinBranch(nn.Module):
    """Trainable GNN over AlphaFold residue graphs; each unique protein is encoded once per batch."""

    def __init__(self, graphs: dict[str, Graph], encoder: ProteinStructureEncoder) -> None:
        super().__init__()
        self.graphs = graphs
        self.encoder = encoder

    def _unique(self, protein_keys: list[str]) -> tuple[GraphBatch, torch.Tensor]:
        unique_keys = sorted(set(protein_keys))
        device = next(self.encoder.parameters()).device
        batch = batch_graphs([self.graphs[key] for key in unique_keys]).to(device)
        position = {key: i for i, key in enumerate(unique_keys)}
        return batch, torch.tensor([position[key] for key in protein_keys], device=device)

    def forward(self, protein_keys: list[str]) -> torch.Tensor:
        batch, row_index = self._unique(protein_keys)
        return self.encoder(batch)[row_index]

    def residue_tokens(self, protein_keys: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch, row_index = self._unique(protein_keys)
        nodes = self.encoder.encoder.node_embeddings(batch)
        dense, mask = to_dense_batch(nodes, batch.batch, batch.num_graphs)
        return dense[row_index], mask[row_index]


def compute_esm_embeddings(
    sequence_by_key: dict[str, str],
    model_name: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoder = ESM2ProteinEncoder(model_name=model_name, freeze_backbone=True, max_length=max_length).to(device)
    encoder.eval()

    embeddings: dict[str, torch.Tensor] = {}
    items = sorted(sequence_by_key.items())
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            pooled = encoder.pooled_embeddings([sequence for _, sequence in batch]).detach().cpu()
            for (key, _), embedding in zip(batch, pooled):
                embeddings[key] = embedding
    return embeddings


def build_protein_graphs(
    rows: list[dict[str, str]], args: argparse.Namespace, device: torch.device | str = "cpu"
) -> dict[str, Graph]:
    builder = ProteinStructureGraphBuilder(
        ProteinGraphConfig(contact_cutoff=args.contact_cutoff, min_plddt=args.min_plddt)
    )
    graphs: dict[str, Graph] = {}
    for row in rows:
        key = row["target_chembl_id"]
        if key in graphs:
            continue
        pdb_path = resolve_path(row["alphafold_pdb_path"])
        if not pdb_path.exists():
            raise SystemExit(f"AlphaFold structure not found for {key}: {pdb_path}")
        graphs[key] = builder.from_pdb(pdb_path, sequence_override=row["protein_sequence"]).to(device)
    return graphs


def build_protein_branch(
    rows: list[dict[str, str]], args: argparse.Namespace, device: torch.device
) -> nn.Module | None:
    if args.protein_encoder == "none":
        return None
    if args.protein_encoder == "esm2":
        sequence_by_key = {row["target_chembl_id"]: row["protein_sequence"] for row in rows}
        try:
            embeddings = compute_esm_embeddings(
                sequence_by_key, args.model_name, args.max_length, args.protein_batch_size, device
            )
        except OSError as exc:
            raise SystemExit(
                "Could not load the ESM-2 pretrained model. The first run needs access to "
                "Hugging Face to download `facebook/esm2_t6_8M_UR50D`, or you must pass "
                "`--model-name /path/to/local/esm2_model_directory`.\n"
                f"Original error: {exc}"
            ) from exc
        return PrecomputedProteinBranch(embeddings, args.embedding_dim)

    encoder = ProteinStructureEncoder(
        graph_config=ProteinGraphConfig(contact_cutoff=args.contact_cutoff, min_plddt=args.min_plddt),
        hidden_dim=args.gnn_hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.protein_gnn_layers,
        dropout=args.dropout,
    )
    return StructureProteinBranch(build_protein_graphs(rows, args, device), encoder)


class MoleculeProteinDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        molecule_inputs: list[object],
        indices: np.ndarray,
    ) -> None:
        self.rows = rows
        self.molecule_inputs = molecule_inputs
        self.indices = indices.tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_idx = self.indices[index]
        row = self.rows[row_idx]
        return {
            "molecule": self.molecule_inputs[row_idx],
            "protein_key": row["target_chembl_id"],
            "label": float(row["label"]),
            "row_idx": row_idx,
        }


class ProteinBucketBatchSampler(torch.utils.data.Sampler):
    """Batches whose rows come from at most ``proteins_per_batch`` proteins.

    With a trainable structure GNN every batch encodes the residue graphs of all
    proteins it touches; on Davis (433 constructs) a random batch touches ~110 of
    them. Bucketing caps that, while rows are still shuffled within each group of
    proteins and the protein order changes every epoch.
    """

    def __init__(
        self, protein_keys: list[str], proteins_per_batch: int, batch_size: int, shuffle: bool, seed: int
    ) -> None:
        self.groups: dict[str, list[int]] = defaultdict(list)
        for position, key in enumerate(protein_keys):
            self.groups[key].append(position)
        self.proteins_per_batch = proteins_per_batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        keys = sorted(self.groups)
        if self.shuffle:
            rng.shuffle(keys)
        batches = []
        for start in range(0, len(keys), self.proteins_per_batch):
            chunk = [position for key in keys[start : start + self.proteins_per_batch] for position in self.groups[key]]
            if self.shuffle:
                rng.shuffle(chunk)
            batches.extend(chunk[i : i + self.batch_size] for i in range(0, len(chunk), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._batches()
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self._batches())


def make_loader(
    rows: list[dict[str, str]],
    molecule_inputs: list[object],
    indices: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = MoleculeProteinDataset(rows, molecule_inputs, indices)
    if getattr(args, "proteins_per_batch", 0) > 0:
        sampler = ProteinBucketBatchSampler(
            [rows[i]["target_chembl_id"] for i in indices.tolist()],
            args.proteins_per_batch,
            args.batch_size,
            shuffle,
            args.seed,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_batch)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate_batch)


def collate_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    molecules = [item["molecule"] for item in batch]
    return {
        "molecule": batch_graphs(molecules) if isinstance(molecules[0], Graph) else torch.stack(molecules),
        "protein_keys": [str(item["protein_key"]) for item in batch],
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "row_idx": torch.tensor([int(item["row_idx"]) for item in batch], dtype=torch.long),
    }


class MoleculeProteinClassifier(nn.Module):
    def __init__(
        self,
        molecule_encoder: nn.Module,
        protein_branch: nn.Module | None,
        fusion: str,
        embedding_dim: int,
        gnn_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        self.molecule_encoder = molecule_encoder
        self.protein_branch = protein_branch
        self.fusion_type = fusion if protein_branch is not None else "concat"
        if self.fusion_type == "cross_attention":
            if not isinstance(molecule_encoder, MoleculeGNNEncoder) or not isinstance(
                protein_branch, StructureProteinBranch
            ):
                raise SystemExit("--fusion cross_attention needs --molecule-encoder gnn --protein-encoder structure_gnn")
            self.fusion = CrossAttentionFusion(
                atom_dim=gnn_hidden_dim,
                residue_dim=gnn_hidden_dim,
                attention_dim=gnn_hidden_dim,
                num_heads=attention_heads,
                hidden_dim=fusion_hidden_dim,
                output_dim=1,
                dropout=dropout,
            )
        elif self.fusion_type == "gated":
            self.fusion = GatedFusion(
                {"molecule": embedding_dim, "protein": embedding_dim},
                fused_dim=embedding_dim,
                hidden_dim=fusion_hidden_dim,
                output_dim=1,
                dropout=dropout,
            )
        else:
            self.fusion = TwoModalityFusion(
                molecule_dim=embedding_dim,
                protein_dim=embedding_dim,
                hidden_dim=fusion_hidden_dim,
                output_dim=1,
                dropout=dropout,
                use_molecule=True,
                use_protein=protein_branch is not None,
                use_image=False,
            )

    def forward(
        self, molecule_input: object, protein_keys: list[str], return_details: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        details: dict[str, torch.Tensor] = {}
        if self.fusion_type == "cross_attention":
            atom_nodes = self.molecule_encoder.encoder.node_embeddings(molecule_input)
            atom_tokens, atom_mask = to_dense_batch(atom_nodes, molecule_input.batch, molecule_input.num_graphs)
            residue_tokens, residue_mask = self.protein_branch.residue_tokens(protein_keys)
            logits, details = self.fusion(
                atom_tokens, atom_mask, residue_tokens, residue_mask, return_attention=True
            )
            details["residue_mask"] = residue_mask
        elif self.fusion_type == "gated":
            logits, gates = self.fusion(
                {
                    "molecule": self.molecule_encoder(molecule_input),
                    "protein": self.protein_branch(protein_keys),
                },
                return_gates=True,
            )
            details["modality_gates"] = gates
        else:
            logits = self.fusion(
                molecule_embedding=self.molecule_encoder(molecule_input),
                protein_embedding=None if self.protein_branch is None else self.protein_branch(protein_keys),
            )
        logits = logits.squeeze(-1)
        return (logits, details) if return_details else logits


def build_molecule_encoder(args: argparse.Namespace) -> nn.Module:
    if args.molecule_encoder == "fingerprint":
        return MoleculeFingerprintMLP(
            input_dim=args.fingerprint_bits,
            hidden_dim=args.molecule_hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout,
        )
    return MoleculeGNNEncoder(
        hidden_dim=args.gnn_hidden_dim,
        embedding_dim=args.embedding_dim,
        num_layers=args.molecule_gnn_layers,
        dropout=args.dropout,
    )


def build_model(rows: list[dict[str, str]], args: argparse.Namespace, device: torch.device) -> MoleculeProteinClassifier:
    return MoleculeProteinClassifier(
        molecule_encoder=build_molecule_encoder(args),
        protein_branch=build_protein_branch(rows, args, device),
        fusion=args.fusion,
        embedding_dim=args.embedding_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout=args.dropout,
        attention_heads=args.attention_heads,
    ).to(device)


def evaluate(
    model: MoleculeProteinClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[float] = []
    logits_out: list[float] = []
    row_indices: list[int] = []
    gate_rows: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            logits, details = model(batch["molecule"].to(device), batch["protein_keys"], return_details=True)
            y_true.extend(batch["label"].tolist())
            logits_out.extend(logits.cpu().tolist())
            row_indices.extend(batch["row_idx"].tolist())
            if "modality_gates" in details:
                gate_rows.extend(details["modality_gates"].cpu().tolist())

    y_true_array = np.array(y_true, dtype=np.int32)
    probs = sigmoid(np.array(logits_out, dtype=np.float32))
    metrics = {
        "loss": log_loss(y_true_array.astype(np.float32), probs),
        **binary_classification_metrics(y_true_array, probs),
        "n_examples": int(len(y_true_array)),
        "n_active": int(np.sum(y_true_array == 1)),
        "n_inactive": int(np.sum(y_true_array == 0)),
    }
    if gate_rows:
        mean_gates = np.mean(np.array(gate_rows), axis=0)
        metrics["mean_gate_molecule"] = float(mean_gates[0])
        metrics["mean_gate_protein"] = float(mean_gates[1])
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
                "row_idx",
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
                    "row_idx": prediction["row_idx"],
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
    parser.add_argument("--split", default="random", choices=["random", "scaffold", "cold_target"])
    parser.add_argument("--molecule-encoder", default="fingerprint", choices=["fingerprint", "gnn"])
    parser.add_argument("--protein-encoder", default="esm2", choices=["esm2", "structure_gnn", "none"])
    parser.add_argument("--fusion", default="concat", choices=["concat", "gated", "cross_attention"])
    parser.add_argument("--model-name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--molecule-hidden-dim", type=int, default=512)
    parser.add_argument("--gnn-hidden-dim", type=int, default=128)
    parser.add_argument("--molecule-gnn-layers", type=int, default=4)
    parser.add_argument("--protein-gnn-layers", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--contact-cutoff", type=float, default=8.0, help="C-alpha contact distance in Angstrom.")
    parser.add_argument(
        "--min-plddt", type=float, default=0.0, help="Drop AlphaFold residues below this pLDDT (e.g. 50)."
    )
    parser.add_argument("--fusion-hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--patience", type=int, default=0, help="Stop after this many epochs without val ROC-AUC gain (0 = off)."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--proteins-per-batch",
        type=int,
        default=0,
        help="Draw each batch from at most this many proteins (0 = random batches). "
        "Speeds up structure_gnn / cross_attention on many-protein data such as Davis (try 16).",
    )
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
    train_idx, val_idx, test_idx, split_diagnostics = make_split(rows, args)

    device = select_device(args.device)
    print(f"Rows: {len(rows)}")
    print(f"Active/inactive: {int(np.sum(labels == 1))}/{int(np.sum(labels == 0))}")
    print(f"Protein constructs: {len({row['target_chembl_id'] for row in rows})}")
    print(f"Split: {args.split} train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Test scaffolds seen in train: {split_diagnostics['scaffolds']['test_seen_in_train']}")
    print(f"Test proteins seen in train: {split_diagnostics['uniprot_ids']['test_seen_in_train']}")
    print(f"Device: {device}")

    print(f"Building molecule inputs ({args.molecule_encoder})")
    molecule_inputs = build_molecule_inputs(rows, args)
    print(f"Building protein branch ({args.protein_encoder}), fusion={args.fusion}")
    model = build_model(rows, args, device)

    loaders = {
        name: make_loader(rows, molecule_inputs, indices, args, shuffle=(name == "train"))
        for name, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    tracker = BestEpochTracker(patience=args.patience)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            logits = model(batch["molecule"].to(device), batch["protein_keys"])
            loss = loss_fn(logits, batch["label"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics, _ = evaluate(model, loaders["val"], device)
        improved = tracker.update(epoch, val_metrics["roc_auc"], model)
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
            f"val_pr_auc={val_metrics['pr_auc']:.4f}" + (" *" if improved else ""),
            flush=True,
        )
        if tracker.should_stop:
            print(f"Early stopping: no val ROC-AUC gain for {args.patience} epochs", flush=True)
            break

    tracker.restore(model)
    print(f"Restored best epoch {tracker.best_epoch} (val ROC-AUC {tracker.best_score:.4f})")
    train_metrics, _ = evaluate(model, loaders["train"], device)
    val_metrics, _ = evaluate(model, loaders["val"], device)
    test_metrics, test_predictions = evaluate(model, loaders["test"], device)

    metrics = {
        "task": "molecule_protein_activity_classification",
        "input_csv": str(args.data),
        "model": f"{args.molecule_encoder} molecule encoder + {args.protein_encoder} protein encoder "
        f"+ {model.fusion_type} fusion",
        "molecule_encoder": args.molecule_encoder,
        "protein_encoder": args.protein_encoder,
        "fusion": model.fusion_type,
        "model_name": args.model_name if args.protein_encoder == "esm2" else None,
        "device": str(device),
        "n_rows": len(rows),
        "n_active": int(np.sum(labels == 1)),
        "n_inactive": int(np.sum(labels == 0)),
        "n_unique_molecules": len({row["curated_smiles"] for row in rows}),
        "n_unique_targets": len({row["target_chembl_id"] for row in rows}),
        "n_unique_uniprot_ids": len({row["uniprot_id"] for row in rows}),
        "split": split_diagnostics,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "model_selection": tracker.summary() | {"epochs_run": len(history)},
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
    if "mean_gate_protein" in test_metrics:
        print(f"Mean modality gates: molecule={test_metrics['mean_gate_molecule']:.3f} "
              f"protein={test_metrics['mean_gate_protein']:.3f}")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote model: {model_path}")


if __name__ == "__main__":
    main()
