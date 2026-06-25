"""Base class for model plugins."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch


class ModelPlugin(torch.nn.Module):
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model plugin name."""

    @abstractmethod
    def forward(self, batch: Any, task: str | None = None) -> Any:
        """Run a task-aware forward pass."""

    @abstractmethod
    def encode(self, batch: Any) -> torch.Tensor:
        """Encode a batch into embeddings."""

    def get_embeddings(self, batch: Any) -> torch.Tensor:
        return self.encode(batch)


__all__ = ["ModelPlugin"]

