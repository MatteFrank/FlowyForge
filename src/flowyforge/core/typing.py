"""Shared lightweight typing placeholders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class ObjectGroup:
    """A named collection of per-object features for one batch."""

    name: str
    features: Any
    mask: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventBatch:
    """Minimal event-batch container shared by data, tasks, and models."""

    objects: Mapping[str, ObjectGroup] = field(default_factory=dict)
    targets: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = ["EventBatch", "ObjectGroup"]

