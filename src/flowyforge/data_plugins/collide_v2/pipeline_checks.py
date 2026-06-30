"""Small helpers for checking COLLIDE pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def file_exists(path: str | Path) -> bool:
    return Path(path).expanduser().is_file()


def collect_pipeline_artifacts(processed_data_dir: str | Path) -> dict[str, bool]:
    root = Path(processed_data_dir).expanduser()
    paths = {
        "vectorized_x": root / "vectorized" / "X.npy",
        "vectorized_y": root / "vectorized" / "y.npy",
        "feature_map": root / "vectorized" / "feature_map.json",
        "vectorization_manifest": root / "vectorized" / "vectorization_manifest.json",
        "preprocessed_x": root / "preprocessed" / "X_preprocessed.npy",
        "preprocessed_y": root / "preprocessed" / "y.npy",
        "preprocessing_manifest": root / "preprocessed" / "preprocessing_manifest.json",
        "training_checkpoint": root / "training" / "classification_mlp" / "checkpoint.pt",
        "training_metrics": root / "training" / "classification_mlp" / "metrics.json",
        "evaluation_metrics": root / "evaluation" / "classification_mlp" / "evaluation_metrics.json",
    }
    return {name: file_exists(path) for name, path in paths.items()}


def count_classes_from_y(path_to_y: str | Path) -> int:
    y = np.load(Path(path_to_y).expanduser())
    return int(np.unique(y).size)


def classification_ready(processed_data_dir: str | Path) -> bool:
    y_path = Path(processed_data_dir).expanduser() / "preprocessed" / "y.npy"
    if not y_path.is_file():
        return False
    return count_classes_from_y(y_path) >= 2


__all__ = [
    "classification_ready",
    "collect_pipeline_artifacts",
    "count_classes_from_y",
    "file_exists",
]
