"""Plotting helper placeholders."""

from __future__ import annotations

from typing import Any


def quick_plot(data: Any, title: str | None = None) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_title(title or "Quick plot")
    ax.plot(data)
    return fig


__all__ = ["quick_plot"]

