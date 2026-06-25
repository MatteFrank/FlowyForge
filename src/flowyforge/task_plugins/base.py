"""Base class for task plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TaskPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Stable task plugin name."""

    @abstractmethod
    def prepare_batch(self, batch: Any) -> Any:
        """Prepare a raw batch for the task."""

    @abstractmethod
    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        """Compute a training loss."""

    @abstractmethod
    def compute_metrics(self, outputs: Any, batch: Any) -> dict[str, float]:
        """Compute task metrics."""


__all__ = ["TaskPlugin"]

