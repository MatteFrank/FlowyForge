import json
from pathlib import Path

import numpy as np
import pytest
import torch

from flowyforge.model_plugins.mlp import MLPModel
from flowyforge.training.evaluation import (
    EvaluationConfig,
    compute_classification_metrics,
    compute_confusion_matrix,
    evaluate_trained_classifier,
)


def _evaluation_config(**overrides: object) -> EvaluationConfig:
    values = {
        "task": "classification",
        "model": "mlp",
        "input_subdir": "preprocessed",
        "training_subdir": "training/classification_mlp",
        "output_subdir": "evaluation/classification_mlp",
        "split": "test",
        "batch_size": 8,
        "device": "cpu",
    }
    values.update(overrides)
    return EvaluationConfig(**values)


def _write_preprocessed_and_checkpoint(processed_dir: Path) -> None:
    preprocessed_dir = processed_dir / "preprocessed"
    training_dir = processed_dir / "training" / "classification_mlp"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    X = rng.normal(size=(30, 4)).astype(np.float32)
    y = np.asarray([0, 1] * 15, dtype=np.int64)
    np.save(preprocessed_dir / "X_preprocessed.npy", X)
    np.save(preprocessed_dir / "y.npy", y)
    (preprocessed_dir / "feature_map.json").write_text(
        json.dumps({f"f{index}": index for index in range(4)}) + "\n",
        encoding="utf-8",
    )
    (preprocessed_dir / "label_map.json").write_text(
        json.dumps({"0": 0, "1": 1}) + "\n",
        encoding="utf-8",
    )

    model = MLPModel(input_dim=4, hidden_dim=8, output_dim=2)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 4,
        "hidden_dim": 8,
        "output_dim": 2,
        "feature_map": {f"f{index}": index for index in range(4)},
        "label_map": {"0": 0, "1": 1},
        "training_config": {
            "task": "classification",
            "model": "mlp",
            "input_subdir": "preprocessed",
            "output_subdir": "training/classification_mlp",
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "hidden_dim": 8,
            "dropout": 0.0,
            "val_fraction": 0.2,
            "test_fraction": 0.2,
            "seed": 42,
            "device": "cpu",
        },
    }
    torch.save(checkpoint, training_dir / "checkpoint.pt")


def test_evaluate_trained_classifier_writes_artifacts(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    _write_preprocessed_and_checkpoint(processed_dir)

    result = evaluate_trained_classifier(processed_dir, _evaluation_config(split="test"))

    assert result.metrics_path.exists()
    assert result.confusion_matrix_path.exists()
    assert result.predictions_path.exists()
    assert result.report_path.exists()
    assert result.split == "test"
    assert result.n_samples == 6

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert "accuracy" in metrics
    assert metrics["n_samples"] == 6
    assert metrics["n_classes"] == 2
    assert metrics["checkpoint_path"].endswith("checkpoint.pt")

    confusion = json.loads(result.confusion_matrix_path.read_text(encoding="utf-8"))
    assert confusion["convention"] == "rows=true, columns=predicted"
    assert len(confusion["matrix"]) == 2

    csv_text = result.predictions_path.read_text(encoding="utf-8")
    assert csv_text.startswith("index,y_true,y_pred,correct")


def test_evaluate_trained_classifier_all_split(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    _write_preprocessed_and_checkpoint(processed_dir)

    result = evaluate_trained_classifier(processed_dir, _evaluation_config(split="all"))

    assert result.n_samples == 30
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["split"] == "all"
    assert metrics["n_samples"] == 30


def test_evaluate_missing_checkpoint_raises(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    preprocessed_dir = processed_dir / "preprocessed"
    preprocessed_dir.mkdir(parents=True)
    np.save(preprocessed_dir / "X_preprocessed.npy", np.ones((4, 2), dtype=np.float32))
    np.save(preprocessed_dir / "y.npy", np.asarray([0, 1, 0, 1], dtype=np.int64))

    with pytest.raises(FileNotFoundError, match="No checkpoint.pt found"):
        evaluate_trained_classifier(processed_dir, _evaluation_config())


def test_compute_metrics_and_confusion_matrix() -> None:
    y_true = np.asarray([0, 0, 1, 1, 1])
    y_pred = np.asarray([0, 1, 1, 1, 0])

    metrics = compute_classification_metrics(y_true, y_pred, n_classes=2)
    matrix = compute_confusion_matrix(y_true, y_pred, n_classes=2)

    assert metrics["accuracy"] == 0.6
    assert metrics["macro_accuracy"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert matrix == [[1, 1], [1, 2]]

