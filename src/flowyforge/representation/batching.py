"""Batching placeholders."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flowyforge.core.typing import EventBatch


def collate_events(events: Sequence[Any]) -> EventBatch:
    return EventBatch(metadata={"num_events": len(events)})


__all__ = ["collate_events"]

