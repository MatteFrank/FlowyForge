"""Graph construction placeholders."""

from __future__ import annotations

from typing import Any


def build_event_graph(batch: Any) -> Any:
    raise NotImplementedError("Graph construction is not implemented in the skeleton.")


__all__ = ["build_event_graph"]

