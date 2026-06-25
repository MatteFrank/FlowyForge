"""Dataset inspection helper placeholders."""

from __future__ import annotations

from typing import Any


def summarize_mapping(config: dict[str, Any]) -> dict[str, str]:
    return {key: type(value).__name__ for key, value in config.items()}


__all__ = ["summarize_mapping"]

