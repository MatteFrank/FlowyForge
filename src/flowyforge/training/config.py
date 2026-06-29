"""Training configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    task: str
    model: str
    input_subdir: str
    output_subdir: str
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    val_fraction: float
    test_fraction: float
    seed: int
    device: str


def training_config_from_dict(
    cfg: dict[str, Any],
    task_override: str | None = None,
    model_override: str | None = None,
) -> TrainingConfig:
    """Build training config from the lightweight project config."""

    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("Config key 'training' must be a mapping when present.")

    task = task_override or training_cfg.get("task", "classification")
    model = model_override or training_cfg.get("model", "mlp")
    return TrainingConfig(
        task=str(task),
        model=str(model),
        input_subdir=str(training_cfg.get("input_subdir", "preprocessed")),
        output_subdir=str(training_cfg.get("output_subdir", "training/classification_mlp")),
        batch_size=int(training_cfg.get("batch_size", 16)),
        epochs=int(training_cfg.get("epochs", 20)),
        learning_rate=float(training_cfg.get("learning_rate", 1.0e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        hidden_dim=int(training_cfg.get("hidden_dim", 32)),
        dropout=float(training_cfg.get("dropout", 0.0)),
        val_fraction=float(training_cfg.get("val_fraction", 0.2)),
        test_fraction=float(training_cfg.get("test_fraction", 0.2)),
        seed=int(training_cfg.get("seed", 42)),
        device=str(training_cfg.get("device", "auto")),
    )


__all__ = ["TrainingConfig", "training_config_from_dict"]

