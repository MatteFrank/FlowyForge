"""Trigger task placeholder."""

from __future__ import annotations

from typing import Any

from flowyforge.task_plugins.base import TaskPlugin


class TriggerTask(TaskPlugin):
    @property
    def name(self) -> str:
        return "trigger"

    def prepare_batch(self, batch: Any) -> Any:
        return batch

    def compute_loss(self, outputs: Any, batch: Any) -> Any:
        raise NotImplementedError("Trigger loss is not implemented in the skeleton.")

    def compute_metrics(self, outputs: Any, batch: Any) -> dict[str, float]:
        return {}


__all__ = ["TriggerTask"]

