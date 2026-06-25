"""Path helpers for COLLIDE-2V data locations."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CollideV2Paths:
    """Describe a COLLIDE-2V data root without touching the filesystem."""

    base_path: str
    backend: str = "local"

    @property
    def root(self) -> Path:
        return Path(self.base_path)

    def describe(self) -> dict[str, str]:
        return {"backend": self.backend, "base_path": self.base_path}


def resolve_base_path(config: dict[str, Any]) -> Path:
    """Resolve the configured parquet dataset root from a project config.

    The path is returned as a normal filesystem path. EOS paths are intentionally
    treated the same way for now so the code can run on LXPLUS without an EOS
    Python dependency.
    """

    paths_config = config.get("paths", {})
    if paths_config and not isinstance(paths_config, dict):
        raise ValueError("Config key 'paths' must be a mapping.")
    if isinstance(paths_config, dict) and paths_config.get("dataset_dir"):
        return Path(os.path.expandvars(str(paths_config["dataset_dir"]))).expanduser()

    data_config = config.get("data", config)
    if not isinstance(data_config, dict):
        raise ValueError("Config must contain a mapping at key 'data'.")

    base_path = data_config.get("base_path")
    if not base_path:
        raise ValueError("Config is missing required key: data.base_path")
    return Path(os.path.expandvars(str(base_path))).expanduser()


def list_parquet_files(base_path: str | Path, max_files: int | None = None) -> list[Path]:
    """Return sorted parquet files under ``base_path``.

    ``base_path`` may be a directory or a single parquet file. Directory paths
    are searched recursively.
    """

    root = Path(base_path).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Base path does not exist: {root}")
    if max_files is not None and max_files < 0:
        raise ValueError("max_files must be non-negative or None.")

    if root.is_file():
        parquet_files = [root] if root.suffix == ".parquet" else []
    else:
        parquet_files = sorted(root.rglob("*.parquet"))

    if max_files is not None:
        return parquet_files[:max_files]
    return parquet_files


__all__ = ["CollideV2Paths", "list_parquet_files", "resolve_base_path"]
