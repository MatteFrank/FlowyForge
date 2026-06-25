"""Classification task placeholder."""

from __future__ import annotations

from typing import Any

from flowyforge.task_plugins.base import TaskPlugin


class ClassificationTask(TaskPlugin):
    @property
    def name(self) -> str:
        return "classification"

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        raise NotImplementedError("Classification loss is not implemented in the skeleton.")

    def compute_metrics(self, outputs: Any, batch: Any) -> dict[str, float]:
        return {}


__all__ = ["ClassificationTask"]

