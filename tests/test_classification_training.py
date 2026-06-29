import json
from pathlib import Path

import numpy as np
import pytest

from flowyforge.training.classification_trainer import train_tiny_mlp_classifier
from flowyforge.training.config import TrainingConfig, training_config_from_dict


def _training_config(**overrides: object) -> TrainingConfig:
    values = {
        "task": "classification",
        "model": "mlp",
        "input_subdir": "preprocessed",
        "output_subdir": "training/classification_mlp",
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "hidden_dim": 8,
        "dropout": 0.0,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "seed": 123,
        "device": "cpu",
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _write_preprocessed(
    processed_dir: Path,
    X: np.ndarray,
    y: np.ndarray | None,
) -> None:
    preprocessed_dir = processed_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    np.save(preprocessed_dir / "X_preprocessed.npy", X.astype(np.float32))
    if y is not None:
        np.save(preprocessed_dir / "y.npy", y.astype(np.int64))
    (preprocessed_dir / "feature_map.json").write_text(
        json.dumps({f"f{index}": index for index in range(X.shape[1])}) + "\n",
        encoding="utf-8",
    )
    (preprocessed_dir / "label_map.json").write_text(
        json.dumps({"0": 0, "1": 1}) + "\n",
        encoding="utf-8",
    )


def test_train_tiny_mlp_classifier_writes_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = np.asarray([0, 1] * 15, dtype=np.int64)
    processed_dir = tmp_path / "processed"
    _write_preprocessed(processed_dir, X, y)

    result = train_tiny_mlp_classifier(processed_dir, _training_config())

    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    assert result.manifest_path.exists()
    assert result.n_features == 4
    assert result.n_classes == 2
    assert result.n_train > 0

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "final_train_loss" in metrics
    assert metrics["n_features"] == 4
    assert metrics["n_classes"] == 2
    assert metrics["epochs"] == 2

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["task"] == "classification"
    assert manifest["model"] == "mlp"


def test_train_tiny_mlp_classifier_missing_y_raises(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    _write_preprocessed(processed_dir, np.ones((4, 2), dtype=np.float32), y=None)

    with pytest.raises(FileNotFoundError, match="No y.npy found"):
        train_tiny_mlp_classifier(processed_dir, _training_config())


def test_train_tiny_mlp_classifier_single_class_raises(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    _write_preprocessed(
        processed_dir,
        np.ones((8, 2), dtype=np.float32),
        np.zeros(8, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="at least two classes"):
        train_tiny_mlp_classifier(processed_dir, _training_config())


def test_training_config_from_dict_uses_overrides() -> None:
    config = training_config_from_dict(
        {
            "training": {
                "task": "trigger",
                "model": "other",
                "epochs": 5,
                "device": "cpu",
            }
        },
        task_override="classification",
        model_override="mlp",
    )

    assert config.task == "classification"
    assert config.model == "mlp"
    assert config.epochs == 5
    assert config.device == "cpu"
