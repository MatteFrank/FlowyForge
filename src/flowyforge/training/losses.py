"""Loss placeholders."""

from __future__ import annotations


def loss_not_implemented() -> None:
    raise NotImplementedError("Loss functions are not implemented in the skeleton.")


__all__ = ["loss_not_implemented"]

