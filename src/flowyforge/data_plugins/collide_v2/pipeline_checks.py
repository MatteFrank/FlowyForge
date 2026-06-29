"""Small helpers for checking COLLIDE pipeline artifacts."""

from __future__ import annotations

from pathlib import Path


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


__all__ = ["collect_pipeline_artifacts", "file_exists"]

