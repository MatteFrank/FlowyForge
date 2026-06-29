"""Minimal preprocessing for vectorized COLLIDE tiny samples.

This module standardizes already-vectorized ``X.npy`` arrays. It is intended
for local and Hugging Face materialized smoke samples; train-only statistics
and EOS-scale preprocessing are deferred to later pipeline work.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_PREPROCESSING_METHOD = "standardize"
DEFAULT_PREPROCESSING_EPS = 1e-6
DEFAULT_PREPROCESSING_INPUT_SUBDIR = "vectorized"
DEFAULT_PREPROCESSING_OUTPUT_SUBDIR = "preprocessed"


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    method: str
    eps: float
    input_subdir: str
    output_subdir: str
    copy_labels: bool


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    input_dir: Path
    output_dir: Path
    x_input_path: Path
    x_output_path: Path
    y_output_path: Path | None
    stats_path: Path
    manifest_path: Path
    n_rows: int
    n_features: int
    method: str


def preprocessing_config_from_dict(cfg: dict[str, Any]) -> PreprocessingConfig:
    """Build preprocessing config from the lightweight project config."""

    preprocessing_cfg = cfg.get("preprocessing", {})
    if not isinstance(preprocessing_cfg, dict):
        raise ValueError("Config key 'preprocessing' must be a mapping when present.")

    return PreprocessingConfig(
        method=str(preprocessing_cfg.get("method", DEFAULT_PREPROCESSING_METHOD)),
        eps=float(preprocessing_cfg.get("eps", DEFAULT_PREPROCESSING_EPS)),
        input_subdir=str(
            preprocessing_cfg.get("input_subdir", DEFAULT_PREPROCESSING_INPUT_SUBDIR)
        ),
        output_subdir=str(
            preprocessing_cfg.get("output_subdir", DEFAULT_PREPROCESSING_OUTPUT_SUBDIR)
        ),
        copy_labels=bool(preprocessing_cfg.get("copy_labels", True)),
    )


def compute_standardization_stats(X: np.ndarray, eps: float = DEFAULT_PREPROCESSING_EPS) -> dict[str, Any]:
    """Compute per-feature standardization statistics."""

    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError("Cannot compute preprocessing statistics for an empty array.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    array = np.asarray(X, dtype=np.float64)
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    safe_std = np.where(std < eps, 1.0, std)

    return {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "safe_std": safe_std.astype(float).tolist(),
        "eps": float(eps),
        "n_rows": int(array.shape[0]),
        "n_features": int(array.shape[1]),
    }


def apply_standardization(X: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Apply standardization using JSON-serializable stats."""

    mean = np.asarray(stats["mean"], dtype=np.float64)
    safe_std = np.asarray(stats["safe_std"], dtype=np.float64)
    array = np.asarray(X, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature array, got shape {array.shape}.")
    if array.shape[1] != mean.shape[0] or mean.shape != safe_std.shape:
        raise ValueError("Stats dimensionality does not match feature array.")
    return ((array - mean) / safe_std).astype(np.float32)


def preprocess_vectorized_dataset(
    processed_data_dir: str | Path,
    config: PreprocessingConfig,
) -> PreprocessingResult:
    """Preprocess a vectorized tiny dataset under ``processed_data_dir``."""

    if config.method != DEFAULT_PREPROCESSING_METHOD:
        raise ValueError(
            f"Unsupported preprocessing method: {config.method}. "
            f"Only '{DEFAULT_PREPROCESSING_METHOD}' is currently implemented."
        )

    processed_dir = Path(processed_data_dir).expanduser()
    input_dir = processed_dir / config.input_subdir
    output_dir = processed_dir / config.output_subdir
    x_input_path = input_dir / "X.npy"
    if not x_input_path.exists():
        raise FileNotFoundError("No vectorized X.npy found. Run scripts/vectorize_collide.py first.")

    X = np.load(x_input_path)
    stats = compute_standardization_stats(X, eps=config.eps)
    X_preprocessed = apply_standardization(X, stats)

    output_dir.mkdir(parents=True, exist_ok=True)
    x_output_path = output_dir / "X_preprocessed.npy"
    stats_path = output_dir / "preprocessing_stats.json"
    manifest_path = output_dir / "preprocessing_manifest.json"
    np.save(x_output_path, X_preprocessed)
    _write_json(stats, stats_path)

    y_output_path = _copy_if_requested(
        source=input_dir / "y.npy",
        destination=output_dir / "y.npy",
        copy_file=config.copy_labels,
    )
    _copy_optional(input_dir / "feature_map.json", output_dir / "feature_map.json")
    _copy_optional(input_dir / "label_map.json", output_dir / "label_map.json")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "x_input_path": str(x_input_path),
        "x_output_path": str(x_output_path),
        "y_available": y_output_path is not None,
        "y_output_path": str(y_output_path) if y_output_path is not None else None,
        "method": config.method,
        "n_rows": int(X_preprocessed.shape[0]),
        "n_features": int(X_preprocessed.shape[1]),
        "warning": "Minimal preprocessing for local/HF tiny samples; train-only statistics and EOS-scale preprocessing come later.",
    }
    _write_json(manifest, manifest_path)

    return PreprocessingResult(
        input_dir=input_dir,
        output_dir=output_dir,
        x_input_path=x_input_path,
        x_output_path=x_output_path,
        y_output_path=y_output_path,
        stats_path=stats_path,
        manifest_path=manifest_path,
        n_rows=int(X_preprocessed.shape[0]),
        n_features=int(X_preprocessed.shape[1]),
        method=config.method,
    )


def _copy_if_requested(source: Path, destination: Path, copy_file: bool) -> Path | None:
    if not source.exists() or not copy_file:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _copy_optional(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "PreprocessingConfig",
    "PreprocessingResult",
    "apply_standardization",
    "compute_standardization_stats",
    "preprocess_vectorized_dataset",
    "preprocessing_config_from_dict",
]

