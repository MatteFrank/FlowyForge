"""Small generic registry for plugin-like objects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class PluginRegistry:
    """Map string names to arbitrary plugin objects."""

    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        if not name:
            raise ValueError("Registry name must be a non-empty string.")
        if name in self._items:
            raise KeyError(f"Registry entry already exists: {name}")
        self._items[name] = obj

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(self.list()) or "<empty>"
            raise KeyError(f"Unknown registry entry: {name}. Available: {available}") from exc

    def list(self) -> list[str]:
        return sorted(self._items)

    def clear(self) -> None:
        self._items.clear()

    def update(self, items: Iterable[tuple[str, Any]]) -> None:
        for name, obj in items:
            self.register(name, obj)


_DEFAULT_REGISTRY = PluginRegistry()


def register(name: str, obj: Any) -> None:
    _DEFAULT_REGISTRY.register(name, obj)


def get(name: str) -> Any:
    return _DEFAULT_REGISTRY.get(name)


def list() -> list[str]:  # noqa: A001 - keep requested public API name.
    return _DEFAULT_REGISTRY.list()


def clear() -> None:
    _DEFAULT_REGISTRY.clear()


__all__ = ["PluginRegistry", "clear", "get", "list", "register"]

