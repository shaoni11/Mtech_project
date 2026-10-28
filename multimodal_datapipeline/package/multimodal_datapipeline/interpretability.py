"""Attribution helpers linking predictions to atoms, residues and image regions.

- Atoms: gradient x input on molecular graph node features (GNN) or on Morgan
  fingerprint bits mapped back to their centre atoms (fingerprint MLP).
- Residues: gradient x input on AlphaFold residue-graph node features, and/or
  cross-attention residue importance; written into a PDB B-factor column so the
  structure can be coloured in PyMOL (``spectrum b``) or ChimeraX.
- Images: Grad-CAM on the last convolution of a microscopy CNN.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def gradient_x_input(features, gradient) -> np.ndarray:
    """Per-node attribution: sum over feature columns of gradient * input."""
    return (gradient * features).sum(dim=-1).detach().cpu().numpy()


def fingerprint_atom_attribution(
    smiles: str, bit_attribution: np.ndarray, radius: int, n_bits: int
) -> np.ndarray:
    """Spread each Morgan bit's attribution over the atoms of its environment."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    bit_info: dict[int, list[tuple[int, int]]] = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=True, bitInfo=bit_info)
    scores = np.zeros(mol.GetNumAtoms(), dtype=np.float64)
    for bit, environments in bit_info.items():
        for centre, env_radius in environments:
            atoms = {centre}
            if env_radius > 0:
                for bond_index in Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, centre):
                    bond = mol.GetBondWithIdx(bond_index)
                    atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            share = bit_attribution[bit] / (len(atoms) * len(environments))
            for atom in atoms:
                scores[atom] += share
    return scores


def draw_atom_attribution(smiles: str, scores: Sequence[float], path: Path, legend: str = "") -> None:
    """SVG of the molecule with atoms coloured red (supports activity) or blue (against)."""
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    scores = np.asarray(scores, dtype=np.float64)
    scale = float(np.max(np.abs(scores))) or 1.0
    colors, radii = {}, {}
    for atom_index, score in enumerate(scores):
        strength = min(1.0, abs(score) / scale)
        if strength < 0.05:
            continue
        colors[atom_index] = (1.0, 1.0 - strength, 1.0 - strength) if score > 0 else (1.0 - strength, 1.0 - strength, 1.0)
        radii[atom_index] = 0.3 + 0.2 * strength
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=list(colors),
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    path.write_text(drawer.GetDrawingText())


def write_residue_scores_pdb(pdb_in: Path, residue_scores: dict[int, float], pdb_out: Path) -> None:
    """Copy a PDB, replacing B-factors with 0-100 scaled scores keyed by C-alpha order index."""
    values = np.array(list(residue_scores.values()), dtype=np.float64)
    low, high = (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
    span = (high - low) or 1.0
    residue_order: dict[tuple[str, str], int] = {}
    lines = []
    with pdb_in.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith("ATOM"):
                key = (line[21], line[22:27])
                if key not in residue_order:
                    residue_order[key] = len(residue_order)
                index = residue_order[key]
                score = 100.0 * (residue_scores.get(index, low) - low) / span
                line = f"{line[:60]}{score:6.2f}{line[66:]}"
            lines.append(line)
    pdb_out.write_text("".join(lines))


class GradCAM:
    """Grad-CAM for a CNN: weights the target layer's activations by pooled gradients."""

    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image, class_index: int | None = None) -> tuple[np.ndarray, int]:
        import torch
        from torch.nn import functional as F

        self.model.eval()
        image = image.unsqueeze(0) if image.dim() == 3 else image
        logits = self.model(image)
        class_index = int(logits.argmax(dim=-1)) if class_index is None else class_index
        self.model.zero_grad()
        logits[0, class_index].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.cpu().numpy(), class_index

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def microscopy_composite(image: np.ndarray) -> np.ndarray:
    """[3, H, W] DAPI/tubulin/actin -> RGB with actin red, tubulin green, DAPI blue."""
    return np.clip(np.stack([image[2], image[1], image[0]], axis=-1), 0.0, 1.0)
