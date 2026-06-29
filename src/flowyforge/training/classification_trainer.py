"""Minimal PyTorch classification trainer for tiny preprocessed arrays."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from flowyforge.model_plugins.mlp import MLPModel
from flowyforge.training.classification_data import (
    load_preprocessed_classification_arrays,
    make_tensor_dataset,
    make_train_val_test_indices,
)
from flowyforge.training.config import TrainingConfig


@dataclass(frozen=True, slots=True)
class ClassificationTrainingResult:
    output_dir: Path
    checkpoint_path: Path
    metrics_path: Path
    manifest_path: Path
    final_train_loss: float
    val_accuracy: float | None
    test_accuracy: float | None
    n_train: int
    n_val: int
    n_test: int
    n_features: int
    n_classes: int


def train_tiny_mlp_classifier(
    processed_data_dir: str | Path,
    config: TrainingConfig,
) -> ClassificationTrainingResult:
    """Train a tiny MLP classifier on preprocessed arrays."""

    if config.task != "classification":
        raise ValueError("Only task='classification' is supported for tiny training.")
    if config.model != "mlp":
        raise ValueError("Only model='mlp' is supported for tiny training.")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    _set_seed(config.seed)
    processed_dir = Path(processed_data_dir).expanduser()
    arrays = load_preprocessed_classification_arrays(processed_dir, input_subdir=config.input_subdir)
    splits = make_train_val_test_indices(
        arrays.y,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    n_features = int(arrays.X.shape[1])
    n_classes = int(len(np.unique(arrays.y)))

    train_dataset = make_tensor_dataset(arrays.X, arrays.y, splits["train"])
    val_dataset = make_tensor_dataset(arrays.X, arrays.y, splits["val"])
    test_dataset = make_tensor_dataset(arrays.X, arrays.y, splits["test"])

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    device = _resolve_device(config.device)
    model = MLPModel(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        output_dim=n_classes,
        dropout=config.dropout,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    final_train_loss = float("nan")
    for _epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, task=config.task)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            batch_size = int(xb.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_seen += batch_size
        final_train_loss = total_loss / max(total_seen, 1)

    val_accuracy = evaluate_accuracy(model, val_loader, device)
    test_accuracy = evaluate_accuracy(model, test_loader, device)

    output_dir = processed_dir / config.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "training_manifest.json"

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "input_dim": n_features,
        "hidden_dim": config.hidden_dim,
        "output_dim": n_classes,
        "feature_map": arrays.feature_map,
        "label_map": arrays.label_map,
        "training_config": asdict(config),
    }
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "final_train_loss": final_train_loss,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "n_train": int(len(splits["train"])),
        "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
        "n_features": n_features,
        "n_classes": n_classes,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
    }
    _write_json(metrics, metrics_path)

    manifest = {
        "processed_data_dir": str(processed_dir),
        "input_subdir": config.input_subdir,
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "task": config.task,
        "model": config.model,
        "warning": "Minimal tiny-sample classifier; not EOS-scale or physics-validated.",
    }
    _write_json(manifest, manifest_path)

    return ClassificationTrainingResult(
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        final_train_loss=final_train_loss,
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        n_train=int(len(splits["train"])),
        n_val=int(len(splits["val"])),
        n_test=int(len(splits["test"])),
        n_features=n_features,
        n_classes=n_classes,
    )


def evaluate_accuracy(model: torch.nn.Module, dataloader: DataLoader, device: torch.device | str) -> float | None:
    if len(dataloader.dataset) == 0:
        return None
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == yb).sum().item())
            total += int(yb.numel())
    if total == 0:
        return None
    return correct / total


def _resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _write_json(payload: dict[str, object], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = ["ClassificationTrainingResult", "evaluate_accuracy", "train_tiny_mlp_classifier"]

