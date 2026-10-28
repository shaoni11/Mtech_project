"""Structure-aware protein encoder over AlphaFold residue contact graphs.

AlphaFold PDB -> C-alpha residues -> contact graph (C-alpha distance < cutoff)
-> message-passing GNN -> mean pool -> protein embedding.

Node features: amino-acid one-hot + AlphaFold pLDDT confidence (stored in the
PDB B-factor column). Edge features: Gaussian radial basis of the C-alpha
distance + a flag for sequence neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from multimodal_datapipeline.models.graph import Graph, GraphBatch, GraphEncoder


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("Install PyTorch before using protein structure models: pip install torch")


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class ResidueStructure:
    sequence: str
    ca_coords: np.ndarray  # [num_residues, 3]
    plddt: np.ndarray  # [num_residues], 0-100


def parse_alphafold_pdb(path: str | Path) -> ResidueStructure:
    """Read C-alpha atoms of the first model of an AlphaFold PDB file."""
    sequence, coords, plddt = [], [], []
    seen = set()
    with Path(path).open(errors="ignore") as handle:
        for line in handle:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            key = (line[21], line[22:27])  # chain, residue number + insertion code
            if key in seen:
                continue
            seen.add(key)
            sequence.append(AA3_TO_1.get(line[17:20].strip(), "X"))
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            plddt.append(float(line[60:66]))

    if not sequence:
        raise ValueError(f"No C-alpha atoms found in {path}")
    return ResidueStructure(
        sequence="".join(sequence),
        ca_coords=np.asarray(coords, dtype=np.float32),
        plddt=np.asarray(plddt, dtype=np.float32),
    )


@dataclass(frozen=True)
class ProteinGraphConfig:
    contact_cutoff: float = 8.0  # Angstrom, C-alpha to C-alpha
    min_plddt: float = 0.0  # drop residues below this confidence (e.g. 50 removes disordered regions)
    num_rbf: int = 8


class ProteinStructureGraphBuilder:
    """Build residue contact graphs from AlphaFold structures."""

    node_dim = len(AMINO_ACIDS) + 1 + 1  # amino acid one-hot + unknown, pLDDT

    def __init__(self, config: ProteinGraphConfig | None = None) -> None:
        self.config = config or ProteinGraphConfig()

    @property
    def edge_dim(self) -> int:
        return self.config.num_rbf + 1

    def from_structure(self, structure: ResidueStructure, sequence_override: str | None = None) -> Graph:
        """Build the residue graph.

        ``sequence_override`` replaces residue identities when it has the same length
        as the structure (e.g. a point-mutant construct on the wild-type AlphaFold
        model); otherwise the structure's own sequence is used.
        """
        _require_torch()
        keep = structure.plddt >= self.config.min_plddt
        if keep.sum() < 2:
            raise ValueError("Fewer than two residues pass the pLDDT filter.")
        full_sequence = structure.sequence
        if sequence_override and len(sequence_override) == len(full_sequence):
            full_sequence = sequence_override
        residue_ids = np.nonzero(keep)[0]
        coords = structure.ca_coords[keep]
        plddt = structure.plddt[keep]
        sequence = [full_sequence[i] for i in residue_ids]

        node_features = np.zeros((len(sequence), self.node_dim), dtype=np.float32)
        for row, residue in enumerate(sequence):
            node_features[row, AMINO_ACIDS.index(residue) if residue in AMINO_ACIDS else len(AMINO_ACIDS)] = 1.0
        node_features[:, -1] = plddt / 100.0

        deltas = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt((deltas * deltas).sum(-1))
        sequence_neighbors = np.abs(residue_ids[:, None] - residue_ids[None, :]) == 1
        adjacency = (distances < self.config.contact_cutoff) | sequence_neighbors
        np.fill_diagonal(adjacency, False)
        sources, destinations = np.nonzero(adjacency)

        edge_distances = distances[sources, destinations]
        centers = np.linspace(0.0, self.config.contact_cutoff, self.config.num_rbf, dtype=np.float32)
        width = self.config.contact_cutoff / self.config.num_rbf
        rbf = np.exp(-(((edge_distances[:, None] - centers[None, :]) / width) ** 2))
        edge_features = np.concatenate(
            [rbf, sequence_neighbors[sources, destinations][:, None].astype(np.float32)], axis=1
        ).astype(np.float32)

        return Graph(
            x=torch.from_numpy(node_features),
            edge_index=torch.from_numpy(np.stack([sources, destinations]).astype(np.int64)),
            edge_attr=torch.from_numpy(edge_features),
        )

    def from_pdb(self, path: str | Path, sequence_override: str | None = None) -> Graph:
        return self.from_structure(parse_alphafold_pdb(path), sequence_override=sequence_override)


class ProteinStructureEncoder(nn.Module if nn is not None else object):
    """AlphaFold residue contact graph -> GNN -> protein embedding."""

    def __init__(
        self,
        graph_config: ProteinGraphConfig | None = None,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        self.graph_builder = ProteinStructureGraphBuilder(graph_config)
        self.encoder = GraphEncoder(
            node_dim=ProteinStructureGraphBuilder.node_dim,
            edge_dim=self.graph_builder.edge_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, graphs: GraphBatch) -> torch.Tensor:
        return self.encoder(graphs)
