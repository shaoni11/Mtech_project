"""Molecule encoder using RDKit Morgan fingerprints followed by an MLP."""

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
