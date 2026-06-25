"""YAML configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


def load_yaml(path: str | Path) -> ConfigDict:
    """Load a YAML mapping from disk."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ValueError(f"Config path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return loaded


def load_config(path: str | Path) -> ConfigDict:
    """Load the project config at ``path``."""

    return load_yaml(path)


__all__ = ["ConfigDict", "load_config", "load_yaml"]

