#!/usr/bin/env python3
"""Explain a trained Baseline 4 (molecule + protein) model.

For the most confident correctly predicted actives in a run's test set, writes:
    <id>_atoms.svg        molecule with atoms coloured by attribution (red = towards active)
    <id>_residues.pdb     AlphaFold model with B-factor = residue attribution (0-100);
                          colour in PyMOL with ``spectrum b, blue_white_red``
    <id>_residues.csv     top residues by attribution (and cross-attention weight if available)
    explanations.csv      one line per explained pair, incl. modality gates for gated fusion

Attribution is gradient x input. Residue-level output needs ``--protein-encoder
structure_gnn``; cross-attention runs also report attention-based residue importance.

Usage:
    python workflows/interpretability/explain_activity_model.py --run-dir results/baseline_4_gnn_structure_xattn_scaffold
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = PROJECT_ROOT / "package"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import torch  # noqa: E402

from multimodal_datapipeline.interpretability import (  # noqa: E402
    draw_atom_attribution,
    fingerprint_atom_attribution,
    gradient_x_input,
    write_residue_scores_pdb,
)
from multimodal_datapipeline.models.graph import Graph, batch_graphs  # noqa: E402
from multimodal_datapipeline.models.protein_structure_encoder import parse_alphafold_pdb  # noqa: E402
from multimodal_datapipeline.utils.baseline_launcher import load_baseline_module  # noqa: E402
from multimodal_datapipeline.utils.training import select_device  # noqa: E402


B4 = load_baseline_module("baseline_4_molecule_protein")


def load_run(run_dir: Path, device: torch.device):
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    defaults = vars(B4.build_arg_parser().parse_args([]))
    args = argparse.Namespace(**(defaults | checkpoint["args"]))
    args.data = Path(args.data)
    rows = B4.read_rows(args.data, max_examples=args.max_examples)
    model = B4.build_model(rows, args, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return args, rows, model


def select_examples(run_dir: Path, top_k: int) -> list[dict[str, str]]:
    with (run_dir / "test_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if "row_idx" not in predictions[0]:
        raise SystemExit("test_predictions.csv has no row_idx column; re-run training with the current train.py.")
    actives = [p for p in predictions if p["label"] == "1"]
    actives.sort(key=lambda p: float(p["predicted_probability_active"]), reverse=True)
    return actives[:top_k]


def explain_pair(model, args, row, device):
    molecule_input = B4.build_molecule_inputs([row], args)[0]
    if isinstance(molecule_input, Graph):
        molecule = batch_graphs([molecule_input]).to(device)
        molecule.x.requires_grad_(True)
        molecule_features = molecule.x
    else:
        molecule = molecule_input.unsqueeze(0).to(device).requires_grad_(True)
        molecule_features = molecule

    key = row["target_chembl_id"]
    branch = model.protein_branch
    residue_graph = None
    if isinstance(branch, B4.StructureProteinBranch):
        original = branch.graphs[key]
        residue_graph = Graph(original.x.clone().requires_grad_(True), original.edge_index, original.edge_attr)
        branch.graphs[key] = residue_graph
    try:
        logits, details = model(molecule, [key], return_details=True)
        model.zero_grad()
        logits.sum().backward()
    finally:
        if residue_graph is not None:
            branch.graphs[key] = original

    if isinstance(molecule_input, Graph):
        atom_scores = gradient_x_input(molecule.x, molecule.x.grad)
    else:
        bit_scores = gradient_x_input(molecule_features.T, molecule_features.grad.T)
        atom_scores = fingerprint_atom_attribution(
            row["curated_smiles"], bit_scores, args.fingerprint_radius, args.fingerprint_bits
        )
    residue_scores = gradient_x_input(residue_graph.x, residue_graph.x.grad) if residue_graph is not None else None
    attention = None
    if "residue_importance" in details:
        mask = details["residue_mask"][0]
        attention = details["residue_importance"][0][mask].detach().cpu().numpy()
    gates = details["modality_gates"][0].detach().cpu().tolist() if "modality_gates" in details else None
    return float(torch.sigmoid(logits).item()), atom_scores, residue_scores, attention, gates


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a trained Baseline 4 model.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-residues", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    cli = parser.parse_args()
    out_dir = cli.out_dir or cli.run_dir / "explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(cli.device)
    args, rows, model = load_run(cli.run_dir, device)
    summary_rows = []
    for rank, prediction in enumerate(select_examples(cli.run_dir, cli.top_k), start=1):
        row = rows[int(prediction["row_idx"])]
        probability, atom_scores, residue_scores, attention, gates = explain_pair(model, args, row, device)
        stem = f"{rank:02d}_{''.join(ch if ch.isalnum() else '_' for ch in row['target_chembl_id'])}"
        draw_atom_attribution(
            row["curated_smiles"],
            atom_scores,
            out_dir / f"{stem}_atoms.svg",
            legend=f"{row['target_chembl_id']}  p(active)={probability:.2f}",
        )

        top_residues = []
        if residue_scores is not None:
            pdb_path = B4.resolve_path(row["alphafold_pdb_path"])
            structure = parse_alphafold_pdb(pdb_path)
            kept = np.nonzero(structure.plddt >= args.min_plddt)[0]
            sequence = row["protein_sequence"] if len(row["protein_sequence"]) == len(structure.sequence) else structure.sequence
            by_residue = {int(index): float(score) for index, score in zip(kept, residue_scores)}
            write_residue_scores_pdb(pdb_path, by_residue, out_dir / f"{stem}_residues.pdb")
            order = np.argsort(-np.abs(residue_scores))[: cli.top_residues]
            with (out_dir / f"{stem}_residues.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["residue_number", "amino_acid", "attribution", "attention", "plddt"])
                for position in order:
                    index = int(kept[position])
                    writer.writerow(
                        [
                            index + 1,
                            sequence[index],
                            f"{residue_scores[position]:.6f}",
                            f"{attention[position]:.6f}" if attention is not None else "",
                            f"{structure.plddt[index]:.1f}",
                        ]
                    )
                    top_residues.append(f"{sequence[index]}{index + 1}")

        summary_rows.append(
            {
                "rank": rank,
                "target": row["target_chembl_id"],
                "uniprot_id": row["uniprot_id"],
                "smiles": row["curated_smiles"],
                "pchembl_or_pkd": row["median_pchembl"],
                "predicted_probability_active": f"{probability:.4f}",
                "top_residues": " ".join(top_residues[:10]),
                "gate_molecule": f"{gates[0]:.3f}" if gates else "",
                "gate_protein": f"{gates[1]:.3f}" if gates else "",
                "files": stem,
            }
        )
        print(f"{rank:02d} {row['target_chembl_id']} p={probability:.3f} top residues: {' '.join(top_residues[:8])}")

    with (out_dir / "explanations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "explanations.json").open("w") as handle:
        json.dump({"run_dir": str(cli.run_dir), "method": "gradient_x_input", "examples": summary_rows}, handle, indent=2)
    print(f"Wrote {len(summary_rows)} explanations to {out_dir}")


if __name__ == "__main__":
    main()
