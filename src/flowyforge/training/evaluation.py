"""Minimal evaluation utilities for tiny classification baselines."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flowyforge.model_plugins.mlp import MLPModel
from flowyforge.training.classification_data import (
    load_preprocessed_classification_arrays,
    make_train_val_test_indices,
)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    task: str
    model: str
    input_subdir: str
    training_subdir: str
    output_subdir: str
    split: str
    batch_size: int
    device: str


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    output_dir: Path
    metrics_path: Path
    confusion_matrix_path: Path
    predictions_path: Path
    report_path: Path
    accuracy: float | None
    n_samples: int
    split: str


def evaluation_config_from_dict(
    cfg: dict[str, Any],
    task_override: str | None = None,
    model_override: str | None = None,
    split_override: str | None = None,
) -> EvaluationConfig:
    """Build evaluation config from the lightweight project config."""

    evaluation_cfg = cfg.get("evaluation", {})
    if not isinstance(evaluation_cfg, dict):
        raise ValueError("Config key 'evaluation' must be a mapping when present.")

    split = split_override or evaluation_cfg.get("split", "test")
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("Evaluation split must be one of: train, val, test, all.")

    return EvaluationConfig(
        task=str(task_override or evaluation_cfg.get("task", "classification")),
        model=str(model_override or evaluation_cfg.get("model", "mlp")),
        input_subdir=str(evaluation_cfg.get("input_subdir", "preprocessed")),
        training_subdir=str(evaluation_cfg.get("training_subdir", "training/classification_mlp")),
        output_subdir=str(evaluation_cfg.get("output_subdir", "evaluation/classification_mlp")),
        split=str(split),
        batch_size=int(evaluation_cfg.get("batch_size", 64)),
        device=str(evaluation_cfg.get("device", "auto")),
    )


def load_trained_mlp(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[MLPModel, dict[str, Any]]:
    """Load a tiny MLP checkpoint produced by B1.9 training."""

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError("No checkpoint.pt found. Run scripts/train_task.py first.")

    resolved_device = _resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=False)
    training_config = checkpoint.get("training_config", {})
    model = MLPModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        dropout=float(training_config.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def predict_classifier(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if X.ndim != 2:
        raise ValueError(f"Expected 2D X array, got shape {X.shape}.")

    resolved_device = _resolve_device(device)
    dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logits_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in dataloader:
            xb = xb.to(resolved_device)
            logits = model(xb)
            predictions = logits.argmax(dim=1)
            logits_chunks.append(logits.detach().cpu().numpy())
            pred_chunks.append(predictions.detach().cpu().numpy().astype(np.int64))

    if not logits_chunks:
        output_dim = getattr(getattr(model, "head", None), "out_features", 0)
        return np.empty((0, int(output_dim)), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.concatenate(logits_chunks, axis=0), np.concatenate(pred_chunks, axis=0)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")

    n_samples = int(y_true.shape[0])
    if n_samples == 0:
        return {
            "accuracy": None,
            "n_samples": 0,
            "n_classes": int(n_classes),
            "per_class": _empty_per_class(n_classes),
            "macro_accuracy": None,
        }

    correct_mask = y_true == y_pred
    per_class: list[dict[str, int | float | None]] = []
    class_accuracies: list[float] = []
    for class_id in range(n_classes):
        class_mask = y_true == class_id
        support = int(class_mask.sum())
        correct = int((correct_mask & class_mask).sum())
        accuracy = correct / support if support > 0 else None
        if accuracy is not None:
            class_accuracies.append(accuracy)
        per_class.append(
            {
                "class_id": class_id,
                "support": support,
                "correct": correct,
                "accuracy": accuracy,
            }
        )

    return {
        "accuracy": float(correct_mask.mean()),
        "n_samples": n_samples,
        "n_classes": int(n_classes),
        "per_class": per_class,
        "macro_accuracy": float(np.mean(class_accuracies)) if class_accuracies else None,
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> list[list[int]]:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, predicted in zip(y_true, y_pred, strict=True):
        if 0 <= int(true) < n_classes and 0 <= int(predicted) < n_classes:
            matrix[int(true), int(predicted)] += 1
    return matrix.tolist()


def evaluate_trained_classifier(
    processed_data_dir: str | Path,
    config: EvaluationConfig,
) -> EvaluationResult:
    """Evaluate a trained tiny MLP classifier and write report artifacts."""

    if config.task != "classification":
        raise ValueError("Only task='classification' is supported for tiny evaluation.")
    if config.model != "mlp":
        raise ValueError("Only model='mlp' is supported for tiny evaluation.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    processed_dir = Path(processed_data_dir).expanduser()
    arrays = load_preprocessed_classification_arrays(processed_dir, input_subdir=config.input_subdir)
    checkpoint_path = processed_dir / config.training_subdir / "checkpoint.pt"
    model, checkpoint = load_trained_mlp(checkpoint_path, device=config.device)
    training_config = checkpoint.get("training_config", {})
    splits = make_train_val_test_indices(
        arrays.y,
        val_fraction=float(training_config.get("val_fraction", 0.2)),
        test_fraction=float(training_config.get("test_fraction", 0.2)),
        seed=int(training_config.get("seed", 42)),
    )
    selected_indices = _select_indices(splits, config.split, len(arrays.y))
    X_eval = arrays.X[selected_indices]
    y_eval = arrays.y[selected_indices]
    n_classes = int(checkpoint.get("output_dim", len(np.unique(arrays.y))))

    if len(selected_indices) == 0:
        logits = np.empty((0, n_classes), dtype=np.float32)
        y_pred = np.empty((0,), dtype=np.int64)
    else:
        logits, y_pred = predict_classifier(model, X_eval, config.batch_size, config.device)

    metrics = compute_classification_metrics(y_eval, y_pred, n_classes=n_classes)
    confusion_matrix = compute_confusion_matrix(y_eval, y_pred, n_classes=n_classes)

    output_dir = processed_dir / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "evaluation_metrics.json"
    confusion_matrix_path = output_dir / "confusion_matrix.json"
    predictions_path = output_dir / "predictions.csv"
    report_path = output_dir / "evaluation_report.md"

    metrics_payload = {
        "split": config.split,
        "accuracy": metrics["accuracy"],
        "macro_accuracy": metrics["macro_accuracy"],
        "n_samples": metrics["n_samples"],
        "n_classes": metrics["n_classes"],
        "per_class": metrics["per_class"],
        "checkpoint_path": str(checkpoint_path),
        "warning": "Minimal tiny-sample evaluation; not EOS-scale or physics-validated.",
    }
    _write_json(metrics_payload, metrics_path)
    _write_json(
        {
            "matrix": confusion_matrix,
            "convention": "rows=true, columns=predicted",
        },
        confusion_matrix_path,
    )
    _write_predictions_csv(predictions_path, selected_indices, y_eval, y_pred)
    _write_report(report_path, metrics_payload, output_dir, predictions_path, confusion_matrix_path)

    return EvaluationResult(
        output_dir=output_dir,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        predictions_path=predictions_path,
        report_path=report_path,
        accuracy=metrics["accuracy"],
        n_samples=int(metrics["n_samples"]),
        split=config.split,
    )


def with_split(config: EvaluationConfig, split: str | None) -> EvaluationConfig:
    if split is None:
        return config
    return replace(config, split=split)


def _select_indices(
    splits: dict[str, np.ndarray],
    split: str,
    n_rows: int,
) -> np.ndarray:
    if split == "all":
        return np.arange(n_rows, dtype=np.int64)
    if split not in splits:
        raise ValueError("Evaluation split must be one of: train, val, test, all.")
    return splits[split]


def _empty_per_class(n_classes: int) -> list[dict[str, int | None]]:
    return [
        {
            "class_id": class_id,
            "support": 0,
            "correct": 0,
            "accuracy": None,
        }
        for class_id in range(n_classes)
    ]


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _write_predictions_csv(
    path: Path,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "y_true", "y_pred", "correct"])
        writer.writeheader()
        for index, true, predicted in zip(indices, y_true, y_pred, strict=True):
            writer.writerow(
                {
                    "index": int(index),
                    "y_true": int(true),
                    "y_pred": int(predicted),
                    "correct": bool(int(true) == int(predicted)),
                }
            )


def _write_report(
    path: Path,
    metrics: dict[str, Any],
    output_dir: Path,
    predictions_path: Path,
    confusion_matrix_path: Path,
) -> None:
    lines = [
        "# COLLIDE Tiny Classifier Evaluation",
        "",
        f"- Split: {metrics['split']}",
        f"- Samples: {metrics['n_samples']}",
        f"- Accuracy: {_format_optional_float(metrics['accuracy'])}",
        f"- Macro accuracy: {_format_optional_float(metrics['macro_accuracy'])}",
        f"- Output directory: `{output_dir}`",
        f"- Predictions: `{predictions_path}`",
        f"- Confusion matrix: `{confusion_matrix_path}`",
        "",
        "> Minimal tiny-sample evaluation; not EOS-scale or physics-validated.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_optional_float(value: float | None) -> str:
    return "null" if value is None else f"{value:.6f}"


def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "EvaluationConfig",
    "EvaluationResult",
    "compute_classification_metrics",
    "compute_confusion_matrix",
    "evaluate_trained_classifier",
    "evaluation_config_from_dict",
    "load_trained_mlp",
    "predict_classifier",
    "with_split",
]

