"""Sampling placeholders for COLLIDE-2V."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar


T = TypeVar("T")


def take(items: Iterable[T], n: int) -> Iterator[T]:
    return islice(items, n)


__all__ = ["take"]

