import json
from pathlib import Path

import numpy as np
import pytest

from flowyforge.data_plugins.collide_v2.preprocessing import (
    PreprocessingConfig,
    apply_standardization,
    compute_standardization_stats,
    preprocess_vectorized_dataset,
    preprocessing_config_from_dict,
)


def _config(**overrides: object) -> PreprocessingConfig:
    values = {
        "method": "standardize",
        "eps": 1e-6,
        "input_subdir": "vectorized",
        "output_subdir": "preprocessed",
        "copy_labels": True,
    }
    values.update(overrides)
    return PreprocessingConfig(**values)


def test_preprocess_vectorized_dataset_standardizes_and_copies_metadata(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    vectorized_dir = processed_dir / "vectorized"
    vectorized_dir.mkdir(parents=True)
    X = np.asarray(
        [
            [1.0, 5.0, 7.0],
            [2.0, 5.0, 9.0],
            [3.0, 5.0, 11.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 1], dtype=np.int64)
    np.save(vectorized_dir / "X.npy", X)
    np.save(vectorized_dir / "y.npy", y)
    (vectorized_dir / "feature_map.json").write_text('{"a": 0, "constant": 1, "c": 2}\n')
    (vectorized_dir / "label_map.json").write_text('{"background": 0, "signal": 1}\n')

    result = preprocess_vectorized_dataset(processed_dir, _config())

    assert result.x_output_path.exists()
    assert result.y_output_path is not None
    assert result.y_output_path.exists()
    assert result.stats_path.exists()
    assert result.manifest_path.exists()
    assert (result.output_dir / "feature_map.json").exists()
    assert (result.output_dir / "label_map.json").exists()

    X_preprocessed = np.load(result.x_output_path)
    assert X_preprocessed.shape == X.shape
    assert X_preprocessed.dtype == np.float32
    assert np.all(np.isfinite(X_preprocessed))
    assert np.allclose(X_preprocessed[:, [0, 2]].mean(axis=0), [0.0, 0.0], atol=1e-6)
    assert np.allclose(X_preprocessed[:, 1], [0.0, 0.0, 0.0])
    assert np.array_equal(np.load(result.y_output_path), y)

    stats = json.loads(result.stats_path.read_text(encoding="utf-8"))
    assert stats["n_rows"] == 3
    assert stats["n_features"] == 3
    assert stats["safe_std"][1] == 1.0

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["y_available"] is True
    assert manifest["method"] == "standardize"
    assert manifest["n_rows"] == 3
    assert manifest["n_features"] == 3


def test_compute_and_apply_standardization_stats() -> None:
    X = np.asarray([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32)

    stats = compute_standardization_stats(X)
    transformed = apply_standardization(X, stats)

    assert stats["safe_std"][1] == 1.0
    assert transformed.dtype == np.float32
    assert np.all(np.isfinite(transformed))
    assert np.allclose(transformed[:, 0].mean(), 0.0)
    assert np.allclose(transformed[:, 1], [0.0, 0.0])


def test_preprocess_missing_x_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No vectorized X.npy found"):
        preprocess_vectorized_dataset(tmp_path / "processed", _config())


def test_preprocess_unsupported_method_raises(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    vectorized_dir = processed_dir / "vectorized"
    vectorized_dir.mkdir(parents=True)
    np.save(vectorized_dir / "X.npy", np.asarray([[1.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="Unsupported preprocessing method"):
        preprocess_vectorized_dataset(processed_dir, _config(method="minmax"))


def test_preprocessing_config_from_dict_defaults_and_overrides() -> None:
    default_config = preprocessing_config_from_dict({})
    assert default_config == PreprocessingConfig(
        method="standardize",
        eps=1e-6,
        input_subdir="vectorized",
        output_subdir="preprocessed",
        copy_labels=True,
    )

    override_config = preprocessing_config_from_dict(
        {
            "preprocessing": {
                "method": "standardize",
                "eps": 1e-4,
                "input_subdir": "vec",
                "output_subdir": "prep",
                "copy_labels": False,
            }
        }
    )
    assert override_config.eps == 1e-4
    assert override_config.input_subdir == "vec"
    assert override_config.output_subdir == "prep"
    assert override_config.copy_labels is False

