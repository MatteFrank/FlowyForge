"""Trainer placeholder."""

from __future__ import annotations

from typing import Any


class Trainer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def fit(self) -> None:
        raise NotImplementedError("Training is not implemented in the skeleton.")


__all__ = ["Trainer"]

