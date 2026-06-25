"""Tiny MLP placeholder model."""

from __future__ import annotations

from typing import Any

import torch

from flowyforge.model_plugins.base import ModelPlugin


class MLPModel(ModelPlugin):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 8, output_dim: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden_dim, output_dim)

    @property
    def name(self) -> str:
        return "mlp"

    def _as_tensor(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch.float()
        if isinstance(batch, dict) and isinstance(batch.get("x"), torch.Tensor):
            return batch["x"].float()
        return torch.zeros(1, self.input_dim)

    def encode(self, batch: Any) -> torch.Tensor:
        return self.encoder(self._as_tensor(batch))

    def forward(self, batch: Any, task: str | None = None) -> torch.Tensor:
        return self.head(self.encode(batch))


__all__ = ["MLPModel"]

