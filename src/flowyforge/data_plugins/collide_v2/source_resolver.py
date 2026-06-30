"""Resolve COLLIDE-2V dataset sources across local, EOS, and HF backends."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flowyforge.data_plugins.collide_v2.eos_paths import list_parquet_files


SUPPORTED_BACKENDS = {"eos", "local", "hf"}


@dataclass(frozen=True, slots=True)
class ResolvedDatasetSource:
    backend: str
    dataset_dir: Path
    processed_data_dir: Path
    tmp_data_dir: Path
    parquet_files: list[Path]
    process_folders: list[Path]
    hf_dataset_name: str | None = None
    hf_split: str | None = None
    hf_data_dir: str | None = None
    hf_data_dirs: list[str] = field(default_factory=list)
    hf_data_files: str | list[str] | None = None
    local_cache_dir: Path | None = None


def resolve_dataset_source(
    cfg: dict[str, Any],
    allow_missing: bool = False,
) -> ResolvedDatasetSource:
    """Resolve a dataset source from a lightweight YAML config dictionary."""

    paths_cfg, data_cfg = _paths_and_data_config(cfg)
    backend = str(paths_cfg.get("dataset_backend", data_cfg.get("backend", ""))).strip()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported dataset backend: {backend!r}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    dataset_dir_value = paths_cfg.get("dataset_dir", data_cfg.get("base_path"))
    if not dataset_dir_value:
        raise ValueError("Config is missing required dataset path: paths.dataset_dir")

    dataset_dir = _expand_path(dataset_dir_value)
    processed_data_dir = _expand_path(paths_cfg.get("processed_data_dir", "data/processed/collide2v"))
    tmp_data_dir = _expand_path(paths_cfg.get("tmp_data_dir", "data/tmp"))

    hf_dataset_name = paths_cfg.get("hf_dataset_name")
    hf_split = paths_cfg.get("hf_split")
    hf_data_dirs = _normalize_hf_data_dirs(paths_cfg)
    hf_data_dir = hf_data_dirs[0] if hf_data_dirs else None
    hf_data_files = paths_cfg.get("hf_data_files")
    local_cache_dir = _optional_path(paths_cfg.get("local_cache_dir"))

    if backend in {"eos", "local"} and not allow_missing and not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    parquet_files: list[Path] = []
    process_folders: list[Path] = []
    if dataset_dir.exists():
        parquet_files = list_parquet_files(dataset_dir)
        process_folders = _list_process_folders(dataset_dir, parquet_files)
    elif backend == "hf" or allow_missing:
        parquet_files = []
        process_folders = []
    else:
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    return ResolvedDatasetSource(
        backend=backend,
        dataset_dir=dataset_dir,
        processed_data_dir=processed_data_dir,
        tmp_data_dir=tmp_data_dir,
        parquet_files=parquet_files,
        process_folders=process_folders,
        hf_dataset_name=str(hf_dataset_name) if hf_dataset_name else None,
        hf_split=str(hf_split) if hf_split else None,
        hf_data_dir=hf_data_dir,
        hf_data_dirs=hf_data_dirs,
        hf_data_files=hf_data_files,
        local_cache_dir=local_cache_dir,
    )


def _paths_and_data_config(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Config key 'data' must be a mapping when present.")

    paths_cfg = cfg.get("paths")
    if paths_cfg is None:
        paths_cfg = {
            "dataset_backend": data_cfg.get("backend", "local"),
            "dataset_dir": data_cfg.get("base_path"),
            "tmp_data_dir": "data/tmp",
            "processed_data_dir": "data/processed/collide2v",
        }
    if not isinstance(paths_cfg, dict):
        raise ValueError("Config key 'paths' must be a mapping when present.")
    return paths_cfg, data_cfg


def _expand_path(value: str | os.PathLike[str]) -> Path:
    expanded = os.path.expandvars(os.fspath(value))
    return Path(expanded).expanduser()


def _optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    return _expand_path(value)


def _normalize_hf_data_dirs(paths_cfg: dict[str, Any]) -> list[str]:
    plural = paths_cfg.get("hf_data_dirs")
    if plural is None:
        single = paths_cfg.get("hf_data_dir")
        return [str(single)] if single else []
    if isinstance(plural, (str, os.PathLike)):
        return [os.fspath(plural)]
    if not isinstance(plural, list):
        raise ValueError("Config key 'paths.hf_data_dirs' must be a list of strings when present.")
    return [str(item) for item in plural if item]


def _list_process_folders(dataset_dir: Path, parquet_files: list[Path]) -> list[Path]:
    if dataset_dir.is_dir():
        immediate_folders = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
        if immediate_folders:
            return immediate_folders

    inferred = {
        path.parent
        for path in parquet_files
        if path.parent != dataset_dir and path.parent.exists()
    }
    return sorted(inferred)


__all__ = ["ResolvedDatasetSource", "resolve_dataset_source"]
