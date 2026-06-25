"""Tokenization placeholders."""

from __future__ import annotations

from typing import Any


class EventTokenizer:
    def tokenize(self, batch: Any) -> Any:
        raise NotImplementedError("Event tokenization is not implemented in the skeleton.")


__all__ = ["EventTokenizer"]

