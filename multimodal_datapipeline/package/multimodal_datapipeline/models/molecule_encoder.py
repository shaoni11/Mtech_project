"""Molecule encoders: Morgan fingerprint MLP and molecular graph GNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from multimodal_datapipeline.models.graph import Graph, GraphBatch, GraphEncoder, batch_graphs


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("Install PyTorch before using molecule models: pip install torch")


def _require_rdkit():
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install RDKit before using Morgan fingerprints: conda install -c conda-forge rdkit"
        ) from exc
    return Chem, DataStructs, AllChem


@dataclass(frozen=True)
class MorganFingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True


class MorganFingerprintFeaturizer:
    """Convert SMILES strings to Morgan fingerprint arrays."""

    def __init__(self, config: MorganFingerprintConfig | None = None) -> None:
        self.config = config or MorganFingerprintConfig()

    def transform_one(self, smiles: str) -> np.ndarray:
        Chem, DataStructs, AllChem = _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=self.config.radius,
            nBits=self.config.n_bits,
            useChirality=self.config.use_chirality,
        )
        array = np.zeros((self.config.n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def transform(self, smiles_batch: Sequence[str]) -> np.ndarray:
        return np.stack([self.transform_one(smiles) for smiles in smiles_batch]).astype(np.float32)


class MoleculeFingerprintMLP(nn.Module if nn is not None else object):
    """Simple molecule-only baseline: Morgan fingerprint -> MLP embedding."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        _require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.network(fingerprints.float())


class MoleculeEncoder(nn.Module if nn is not None else object):
    """End-to-end SMILES encoder wrapping RDKit fingerprints and an MLP."""

    def __init__(
        self,
        fingerprint_config: MorganFingerprintConfig | None = None,
        hidden_dim: int = 512,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        _require_torch()
        super().__init__()
        self.featurizer = MorganFingerprintFeaturizer(fingerprint_config)
        fp_dim = self.featurizer.config.n_bits
        self.encoder = MoleculeFingerprintMLP(
            input_dim=fp_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def featurize_smiles(self, smiles_batch: Sequence[str], device: torch.device | str | None = None) -> torch.Tensor:
        features = self.featurizer.transform(smiles_batch)
        tensor = torch.from_numpy(features)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def forward(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.encoder(fingerprints)

    def forward_smiles(self, smiles_batch: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        fingerprints = self.featurize_smiles(smiles_batch, device=device)
        return self.forward(fingerprints)


ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
TOTAL_HYDROGENS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
CHIRAL_TAGS = ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]
BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]


def _one_hot(value: object, choices: Sequence[object]) -> list[float]:
    """One-hot over ``choices`` plus a trailing 'other' slot."""
    encoding = [0.0] * (len(choices) + 1)
    encoding[choices.index(value) if value in choices else len(choices)] = 1.0
    return encoding


class MoleculeGraphFeaturizer:
    """Convert SMILES strings to atom/bond graphs for the molecular GNN."""

    node_dim = (
        len(ATOM_SYMBOLS) + 1
        + len(ATOM_DEGREES) + 1
        + len(FORMAL_CHARGES) + 1
        + len(TOTAL_HYDROGENS) + 1
        + len(HYBRIDIZATIONS) + 1
        + len(CHIRAL_TAGS) + 1
        + 2  # aromatic, in ring
    )
    edge_dim = len(BOND_TYPES) + 1 + 3  # bond type, conjugated, in ring, has stereo

    def atom_features(self, atom) -> list[float]:
        return (
            _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
            + _one_hot(atom.GetDegree(), ATOM_DEGREES)
            + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
            + _one_hot(atom.GetTotalNumHs(), TOTAL_HYDROGENS)
            + _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
            + _one_hot(str(atom.GetChiralTag()), CHIRAL_TAGS)
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
        )

    def bond_features(self, bond) -> list[float]:
        return _one_hot(str(bond.GetBondType()), BOND_TYPES) + [
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            float(str(bond.GetStereo()) != "STEREONONE"),
        ]

    def transform_one(self, smiles: str) -> Graph:
        _require_torch()
        Chem, _, _ = _require_rdkit()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        x = torch.tensor([self.atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float32)
        sources, destinations, edge_features = [], [], []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            features = self.bond_features(bond)
            sources += [begin, end]
            destinations += [end, begin]
            edge_features += [features, features]

        edge_index = torch.tensor([sources, destinations], dtype=torch.long).reshape(2, -1)
        edge_attr = torch.tensor(edge_features, dtype=torch.float32).reshape(-1, self.edge_dim)
        return Graph(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def transform(self, smiles_batch: Sequence[str]) -> list[Graph]:
        return [self.transform_one(smiles) for smiles in smiles_batch]


class MoleculeGNNEncoder(nn.Module if nn is not None else object):
    """Molecular graph -> message-passing GNN -> mean pool -> embedding."""

    def __init__(
        self,
        hidden_dim: int = 128,
        embedding_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        _require_torch()
        super().__init__()
        self.featurizer = MoleculeGraphFeaturizer()
        self.encoder = GraphEncoder(
            node_dim=MoleculeGraphFeaturizer.node_dim,
            edge_dim=MoleculeGraphFeaturizer.edge_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, graphs: GraphBatch) -> torch.Tensor:
        return self.encoder(graphs)

    def forward_smiles(self, smiles_batch: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        graphs = batch_graphs(self.featurizer.transform(smiles_batch)).to(device)
        return self.forward(graphs)
