"""Minimal vectorization for tiny COLLIDE parquet samples.

This module is intentionally small and pandas-based. It is suitable for local
and Hugging Face materialized smoke samples, not full EOS-scale production
vectorization. The scalable EOS path will need chunking and batch execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


DEFAULT_EXCLUDE_COLUMNS = ["event_id", "process_name", "view"]
DEFAULT_VECTOR_OUTPUT_SUBDIR = "vectorized"


@dataclass(frozen=True, slots=True)
class VectorizationConfig:
    feature_columns: list[str] | None
    label_column: str | None
    exclude_columns: list[str]
    max_files: int | None
    max_rows: int | None
    output_subdir: str


@dataclass(frozen=True, slots=True)
class VectorizationResult:
    output_dir: Path
    x_path: Path
    y_path: Path | None
    feature_map_path: Path
    label_map_path: Path | None
    manifest_path: Path
    n_rows: int
    n_features: int
    feature_columns: list[str]
    label_column: str | None


def vectorization_config_from_dict(cfg: dict[str, Any]) -> VectorizationConfig:
    """Build a vectorization config from the lightweight project config."""

    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("Config key 'data' must be a mapping when present.")

    feature_columns = data_cfg.get("feature_columns")
    if feature_columns is not None:
        feature_columns = [str(column) for column in feature_columns]

    exclude_columns = data_cfg.get("exclude_columns", DEFAULT_EXCLUDE_COLUMNS)
    return VectorizationConfig(
        feature_columns=feature_columns,
        label_column=data_cfg.get("label_column", "process_id"),
        exclude_columns=[str(column) for column in exclude_columns],
        max_files=data_cfg.get("max_files"),
        max_rows=data_cfg.get("max_rows"),
        output_subdir=str(data_cfg.get("output_subdir", DEFAULT_VECTOR_OUTPUT_SUBDIR)),
    )


def infer_feature_columns(
    df: pd.DataFrame,
    label_column: str | None,
    exclude_columns: list[str],
) -> list[str]:
    """Infer numeric scalar feature columns in stable dataframe order."""

    excluded = set(exclude_columns)
    if label_column is not None:
        excluded.add(label_column)

    columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        if is_numeric_dtype(df[column]):
            columns.append(str(column))
    return columns


def build_feature_map(columns: list[str]) -> dict[str, int]:
    return {column: index for index, column in enumerate(columns)}


def save_feature_map(feature_map: dict[str, int], path: str | Path) -> None:
    _write_json(feature_map, path)


def load_feature_map(path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload.items()}


def encode_labels(values: Any) -> tuple[np.ndarray, dict[str, int]]:
    """Encode labels as deterministic int64 IDs.

    The JSON-safe label map uses string representations of labels as keys.
    """

    label_strings = pd.Series(values).map(_label_to_string).to_numpy(dtype=str)
    unique_labels = sorted(set(label_strings), key=str)
    label_map = {label: index for index, label in enumerate(unique_labels)}
    encoded = np.asarray([label_map[label] for label in label_strings], dtype=np.int64)
    return encoded, label_map


def vectorize_parquet_files(
    parquet_files: list[str | Path],
    output_dir: str | Path,
    config: VectorizationConfig,
) -> VectorizationResult:
    """Vectorize tiny parquet samples into pploner-style ``.npy`` arrays.

    This function reads small parquet files with pandas and concatenates them in
    memory. It is a compatibility step for local/HF tiny samples; full EOS-scale
    vectorization comes later.
    """

    if config.max_files is not None and config.max_files < 0:
        raise ValueError("max_files must be non-negative or None.")
    if config.max_rows is not None and config.max_rows < 0:
        raise ValueError("max_rows must be non-negative or None.")

    selected_files = [Path(path).expanduser() for path in parquet_files]
    if config.max_files is not None:
        selected_files = selected_files[: config.max_files]
    if not selected_files:
        raise ValueError("No parquet files were provided for vectorization.")

    dataframes: list[pd.DataFrame] = []
    input_files: list[str] = []
    remaining_rows = config.max_rows
    feature_columns = list(config.feature_columns) if config.feature_columns is not None else None

    for path in selected_files:
        if remaining_rows is not None and remaining_rows <= 0:
            break

        frame = pd.read_parquet(path)
        if remaining_rows is not None:
            frame = frame.head(remaining_rows)
        if frame.empty:
            continue

        if feature_columns is None:
            feature_columns = infer_feature_columns(frame, config.label_column, config.exclude_columns)
            if not feature_columns:
                raise ValueError(f"Could not infer numeric feature columns from first readable file: {path}")

        _validate_feature_columns(frame, feature_columns, path)
        dataframes.append(frame)
        input_files.append(str(path))

        if remaining_rows is not None:
            remaining_rows -= len(frame)

    if not dataframes:
        raise ValueError("No rows were available for vectorization.")
    if feature_columns is None:
        raise ValueError("No feature columns were configured or inferred.")

    combined = pd.concat(dataframes, ignore_index=True)
    if config.max_rows is not None:
        combined = combined.head(config.max_rows)

    x = combined[feature_columns].to_numpy(dtype=np.float32, copy=True)
    y: np.ndarray | None = None
    label_map: dict[str, int] | None = None
    label_column = config.label_column if config.label_column in combined.columns else None
    if label_column is not None:
        y, label_map = encode_labels(combined[label_column])

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    x_path = output_path / "X.npy"
    y_path = output_path / "y.npy" if y is not None else None
    feature_map_path = output_path / "feature_map.json"
    label_map_path = output_path / "label_map.json" if label_map is not None else None
    manifest_path = output_path / "vectorization_manifest.json"

    np.save(x_path, x)
    if y is not None and y_path is not None:
        np.save(y_path, y)

    feature_map = build_feature_map(feature_columns)
    save_feature_map(feature_map, feature_map_path)
    if label_map is not None and label_map_path is not None:
        _write_json(label_map, label_map_path)

    manifest = {
        "input_files": input_files,
        "output_dir": str(output_path),
        "x_path": str(x_path),
        "y_path": str(y_path) if y_path is not None else None,
        "feature_map_path": str(feature_map_path),
        "label_map_path": str(label_map_path) if label_map_path is not None else None,
        "n_rows": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "feature_columns": feature_columns,
        "label_column": label_column,
        "has_labels": y is not None,
        "warning": "Minimal vectorization for local/HF tiny samples; full EOS-scale vectorization comes later.",
    }
    _write_json(manifest, manifest_path)

    return VectorizationResult(
        output_dir=output_path,
        x_path=x_path,
        y_path=y_path,
        feature_map_path=feature_map_path,
        label_map_path=label_map_path,
        manifest_path=manifest_path,
        n_rows=int(x.shape[0]),
        n_features=int(x.shape[1]),
        feature_columns=feature_columns,
        label_column=label_column,
    )


def _validate_feature_columns(frame: pd.DataFrame, feature_columns: list[str], path: Path) -> None:
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {path}: {missing}")
    non_numeric = [column for column in feature_columns if not is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric in {path}: {non_numeric}")


def _label_to_string(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "VectorizationConfig",
    "VectorizationResult",
    "build_feature_map",
    "encode_labels",
    "infer_feature_columns",
    "load_feature_map",
    "save_feature_map",
    "vectorization_config_from_dict",
    "vectorize_parquet_files",
]

