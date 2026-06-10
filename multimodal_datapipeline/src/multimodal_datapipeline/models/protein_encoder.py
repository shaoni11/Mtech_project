"""Protein sequence encoder using pretrained ESM-2 embeddings."""

from __future__ import annotations

from typing import Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("Install PyTorch before using protein models: pip install torch")


def _load_transformers():
    try:
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install transformers before using ESM-2: pip install transformers"
        ) from exc
    return AutoModel, AutoTokenizer


class ESM2ProteinEncoder(nn.Module if nn is not None else object):
    """Protein sequence -> ESM-2 embedding -> optional projection layer."""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
        max_length: int = 1024,
    ) -> None:
        _require_torch()
        super().__init__()
        AutoModel, AutoTokenizer = _load_transformers()

        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = nn.Linear(hidden_size, embedding_dim)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, sequences: Sequence[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)

        outputs = self.backbone(**tokens)
        residue_embeddings = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(residue_embeddings.dtype)

        # Exclude special tokens from the mean where possible.
        if mask.shape[1] > 2:
            residue_embeddings = residue_embeddings[:, 1:-1, :]
            mask = mask[:, 1:-1, :]

        pooled = (residue_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)


ProteinEncoder = ESM2ProteinEncoder
