"""Feature extraction utilities owned by deep_learning_project."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class MorganFingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True


class MorganFingerprintFeaturizer:
    """Convert SMILES strings to RDKit Morgan fingerprint arrays."""

    def __init__(self, config: MorganFingerprintConfig | None = None) -> None:
        self.config = config or MorganFingerprintConfig()

    def transform_one(self, smiles: str) -> np.ndarray:
        try:
            from rdkit import Chem, DataStructs, RDLogger
            from rdkit.Chem import AllChem
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RDKit is required for Morgan fingerprints. Use the existing project venv "
                "or install rdkit from conda-forge."
            ) from exc

        RDLogger.DisableLog("rdApp.warning")
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


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def protein_kmer_features(sequence: str, dim: int = 1024, k: int = 3) -> np.ndarray:
    """Return L2-normalized hashed amino-acid k-mer features."""
    features = np.zeros(dim, dtype=np.float32)
    sequence = "".join(sequence.split()).upper()
    if len(sequence) < k:
        return features

    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        hashed = stable_hash(kmer)
        sign = 1.0 if hashed % 2 == 0 else -1.0
        features[(hashed // 2) % dim] += sign

    norm = float(np.linalg.norm(features))
    if norm > 0.0:
        features /= norm
    return features


def protein_kmer_feature_matrix(sequences: Sequence[str], dim: int = 1024, k: int = 3) -> np.ndarray:
    return np.stack([protein_kmer_features(sequence, dim=dim, k=k) for sequence in sequences]).astype(np.float32)

