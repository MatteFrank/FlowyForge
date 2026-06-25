"""Report helper placeholders."""

from __future__ import annotations

from typing import Any


def make_text_report(summary: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in summary.items())


__all__ = ["make_text_report"]

