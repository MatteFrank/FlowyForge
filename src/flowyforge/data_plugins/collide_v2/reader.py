"""Reader placeholders for COLLIDE-2V data."""

from __future__ import annotations

from typing import Any


class CollideV2Reader:
    """Minimal reader shell; real IO is added later."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def iter_batches(self) -> Any:
        raise NotImplementedError("COLLIDE-2V reading is not implemented in the skeleton.")


__all__ = ["CollideV2Reader"]

