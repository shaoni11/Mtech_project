"""Phenotypic image encoder using pretrained DINOv2 embeddings."""

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
        raise ModuleNotFoundError("Install PyTorch before using image models: pip install torch")


def _load_transformers():
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Install transformers before using DINOv2: pip install transformers"
        ) from exc
    return AutoImageProcessor, AutoModel


class DINOv2ImageEncoder(nn.Module if nn is not None else object):
    """BBBC021 image -> DINOv2 embedding -> optional projection layer."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        embedding_dim: int = 256,
        freeze_backbone: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        AutoImageProcessor, AutoModel = _load_transformers()

        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.projection = nn.Linear(hidden_size, embedding_dim)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: Sequence[object] | torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device

        if torch.is_tensor(images):
            pixel_values = images.to(device)
        else:
            encoded = self.processor(images=list(images), return_tensors="pt")
            pixel_values = encoded["pixel_values"].to(device)

        outputs = self.backbone(pixel_values=pixel_values)
        if getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        return self.projection(pooled)


ImageEncoder = DINOv2ImageEncoder
