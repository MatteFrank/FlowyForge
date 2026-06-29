"""Array loading and splitting utilities for tiny classification baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True, slots=True)
class ClassificationArrays:
    X: np.ndarray
    y: np.ndarray
    feature_map: dict[str, int] | None
    label_map: dict[str, int] | None


def load_preprocessed_classification_arrays(
    processed_data_dir: str | Path,
    input_subdir: str = "preprocessed",
) -> ClassificationArrays:
    input_dir = Path(processed_data_dir).expanduser() / input_subdir
    x_path = input_dir / "X_preprocessed.npy"
    y_path = input_dir / "y.npy"
    if not x_path.exists():
        raise FileNotFoundError(
            "No preprocessed X_preprocessed.npy found. Run scripts/preprocess_collide.py first."
        )
    if not y_path.exists():
        raise FileNotFoundError(f"No y.npy found for classification training: {y_path}")

    X = np.load(x_path)
    y = np.load(y_path)
    if X.ndim != 2:
        raise ValueError(f"X_preprocessed.npy must be 2D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y.npy must be 1D, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y row counts differ: {X.shape[0]} vs {y.shape[0]}.")
    if len(np.unique(y)) < 2:
        raise ValueError("Classification training requires at least two classes.")

    return ClassificationArrays(
        X=X.astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        feature_map=_load_json_map(input_dir / "feature_map.json"),
        label_map=_load_json_map(input_dir / "label_map.json"),
    )


def make_train_val_test_indices(
    y: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    if y.ndim != 1:
        raise ValueError("y must be 1D.")
    if len(y) == 0:
        raise ValueError("Cannot split an empty dataset.")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be non-negative and sum to < 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(y), dtype=np.int64)
    rng.shuffle(indices)

    n_total = len(indices)
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise ValueError("Train split would be empty.")

    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def make_tensor_dataset(X: np.ndarray, y: np.ndarray, indices: np.ndarray) -> TensorDataset:
    x_tensor = torch.as_tensor(X[indices], dtype=torch.float32)
    y_tensor = torch.as_tensor(y[indices], dtype=torch.long)
    return TensorDataset(x_tensor, y_tensor)


def _load_json_map(path: Path) -> dict[str, int] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload.items()}


__all__ = [
    "ClassificationArrays",
    "load_preprocessed_classification_arrays",
    "make_tensor_dataset",
    "make_train_val_test_indices",
]

