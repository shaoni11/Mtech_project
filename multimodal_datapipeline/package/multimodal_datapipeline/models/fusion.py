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


class GatedFusion(nn.Module if nn is not None else object):
    """Project each modality to a shared space and mix them with learned softmax gates.

    The gates are per example and sum to one across modalities, so they double as
    a readout of how much each modality contributed to a prediction.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        fused_dim: int = 256,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        _require_torch()
        super().__init__()
        if not input_dims:
            raise ValueError("At least one modality must be enabled.")
        self.modalities = list(input_dims)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dim, fused_dim), nn.LayerNorm(fused_dim), nn.ReLU())
                for name, dim in input_dims.items()
            }
        )
        self.gate = nn.Linear(fused_dim * len(self.modalities), len(self.modalities))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        embeddings: dict[str, torch.Tensor],
        return_gates: bool = False,
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``modality_mask`` [B, n_modalities] (True = keep) hides modalities per example;
        a hidden modality gets zero gate weight (modality masking / dropout)."""
        projected = [self.projections[name](embeddings[name]) for name in self.modalities]
        if modality_mask is not None:
            projected = [vector * modality_mask[:, i : i + 1].to(vector.dtype) for i, vector in enumerate(projected)]
        gate_logits = self.gate(torch.cat(projected, dim=-1))
        if modality_mask is not None:
            gate_logits = gate_logits.masked_fill(~modality_mask, float("-inf"))
        gates = torch.softmax(gate_logits, dim=-1)
        fused = sum(gates[:, index : index + 1] * vector for index, vector in enumerate(projected))
        logits = self.head(fused)
        return (logits, gates) if return_gates else logits


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class CrossAttentionFusion(nn.Module if nn is not None else object):
    """Molecule atoms attend over protein residues; optional extra modality vectors.

    atom tokens [B, Na, atom_dim] --query--> residue tokens [B, Nr, residue_dim]
    The atom->residue attention averaged over atoms gives a per-residue importance
    score, which also pools the protein representation.
    """

    def __init__(
        self,
        atom_dim: int,
        residue_dim: int,
        attention_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.2,
        extra_dims: dict[str, int] | None = None,
    ) -> None:
        _require_torch()
        super().__init__()
        self.atom_projection = nn.Linear(atom_dim, attention_dim)
        self.residue_projection = nn.Linear(residue_dim, attention_dim)
        self.attention = nn.MultiheadAttention(attention_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(attention_dim)
        self.extra_names = list(extra_dims or {})
        self.extra_projections = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(dim, attention_dim), nn.ReLU()) for name, dim in (extra_dims or {}).items()}
        )
        self.head = nn.Sequential(
            nn.Linear(attention_dim * (2 + len(self.extra_names)), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        atom_tokens: torch.Tensor,
        atom_mask: torch.Tensor,
        residue_tokens: torch.Tensor,
        residue_mask: torch.Tensor,
        extra: dict[str, torch.Tensor] | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        atoms = self.atom_projection(atom_tokens)
        residues = self.residue_projection(residue_tokens)
        attended, weights = self.attention(
            atoms, residues, residues, key_padding_mask=~residue_mask, need_weights=True
        )
        atoms = self.norm(atoms + attended)
        molecule_context = _masked_mean(atoms, atom_mask)

        # weights: [B, Na, Nr]; average over real atoms -> residue importance [B, Nr].
        atom_weights = atom_mask.unsqueeze(-1).to(weights.dtype)
        residue_importance = (weights * atom_weights).sum(dim=1) / atom_weights.sum(dim=1).clamp_min(1.0)
        protein_context = (residue_importance.unsqueeze(-1) * residues).sum(dim=1)

        parts = [molecule_context, protein_context]
        for name in self.extra_names:
            parts.append(self.extra_projections[name](extra[name]))
        logits = self.head(torch.cat(parts, dim=-1))
        if return_attention:
            return logits, {"atom_residue_attention": weights, "residue_importance": residue_importance}
        return logits


def random_modality_mask(batch_size: int, num_modalities: int, drop_probability: float, device=None) -> torch.Tensor:
    """Keep each modality with probability 1 - p, but always keep at least one per example."""
    keep = torch.rand(batch_size, num_modalities, device=device) >= drop_probability
    empty = ~keep.any(dim=1)
    if empty.any():
        rescue = torch.randint(0, num_modalities, (int(empty.sum()),), device=device)
        keep[empty.nonzero(as_tuple=True)[0], rescue] = True
    return keep
