"""Fusion heads for molecule, protein, and image embeddings."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("Install PyTorch before using fusion models: pip install torch")


class ConcatenationFusion(nn.Module if nn is not None else object):
    """Concatenate available modality embeddings and predict with an MLP."""

    def __init__(
        self,
        molecule_dim: int = 256,
        protein_dim: int = 256,
        image_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 1,
        dropout: float = 0.2,
        use_molecule: bool = True,
        use_protein: bool = True,
        use_image: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        self.use_molecule = use_molecule
        self.use_protein = use_protein
        self.use_image = use_image

        input_dim = 0
        if use_molecule:
            input_dim += molecule_dim
        if use_protein:
            input_dim += protein_dim
        if use_image:
            input_dim += image_dim
        if input_dim == 0:
            raise ValueError("At least one modality must be enabled.")

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(
        self,
        molecule_embedding: torch.Tensor | None = None,
        protein_embedding: torch.Tensor | None = None,
        image_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = []
        if self.use_molecule:
            if molecule_embedding is None:
                raise ValueError("molecule_embedding is required.")
            parts.append(molecule_embedding)
        if self.use_protein:
            if protein_embedding is None:
                raise ValueError("protein_embedding is required.")
            parts.append(protein_embedding)
        if self.use_image:
            if image_embedding is None:
                raise ValueError("image_embedding is required.")
            parts.append(image_embedding)

        fused = torch.cat(parts, dim=-1)
        return self.head(fused)


class TwoModalityFusion(ConcatenationFusion):
    """Convenience wrapper for common two-modality baselines."""

    @classmethod
    def molecule_protein(cls, **kwargs) -> "TwoModalityFusion":
        return cls(use_molecule=True, use_protein=True, use_image=False, **kwargs)

    @classmethod
    def molecule_image(cls, **kwargs) -> "TwoModalityFusion":
        return cls(use_molecule=True, use_protein=False, use_image=True, **kwargs)

    @classmethod
    def protein_image(cls, **kwargs) -> "TwoModalityFusion":
        return cls(use_molecule=False, use_protein=True, use_image=True, **kwargs)


FusionHead = ConcatenationFusion
