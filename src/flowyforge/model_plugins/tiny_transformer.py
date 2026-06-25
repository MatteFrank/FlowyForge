"""Tiny transformer placeholder model."""

from __future__ import annotations

from typing import Any

import torch

from flowyforge.model_plugins.base import ModelPlugin


class TinyTransformerModel(ModelPlugin):
    def __init__(self, embedding_dim: int = 8) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.placeholder = torch.nn.Parameter(torch.zeros(embedding_dim))

    @property
    def name(self) -> str:
        return "tiny_transformer"

    def encode(self, batch: Any) -> torch.Tensor:
        return self.placeholder.unsqueeze(0)

    def forward(self, batch: Any, task: str | None = None) -> torch.Tensor:
        return self.encode(batch)


__all__ = ["TinyTransformerModel"]

