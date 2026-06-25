"""Transform placeholders for COLLIDE-2V."""

from __future__ import annotations

from typing import TypeVar


T = TypeVar("T")


def identity_transform(batch: T) -> T:
    return batch


__all__ = ["identity_transform"]

