"""CPU feasibility training for the structured COLLIDE foundation slice."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from flowyforge.core.typing import EventBatch
from flowyforge.data_plugins.collide_v2.object_dataset import (
    collate_event_batches, collection_feature_dims, default_collection_specs,
    expand_parquet_paths, load_process_events,
)
from flowyforge.model_plugins import EventClassificationModel, FoundationEncoder, PerceiverConfig


def stratified_event_split(events: Sequence[EventBatch], seed: int) -> dict[str, list[int]]:
    """Approximately 60/20/20 per class, with at least one event in each split."""
    labels = np.asarray([int(event.targets.item()) for event in events])
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for label in sorted(set(labels.tolist())):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < 3:
            raise ValueError(f"Class {label} needs at least 3 events for train/validation/test.")
        held_out = max(1, int(len(indices) * 0.2))
        splits["validation"].extend(indices[:held_out].tolist())
        splits["test"].extend(indices[held_out:2 * held_out].tolist())
        splits["train"].extend(indices[2 * held_out:].tolist())
    return {name: rng.permutation(indices).tolist() for name, indices in splits.items()}


def _evaluate(model: EventClassificationModel, loader: DataLoader) -> dict[str, Any]:
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    confusion = torch.zeros(model.num_classes, model.num_classes, dtype=torch.long)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, batch.targets, reduction="sum")
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite logits or evaluation loss.")
            predicted = logits.argmax(dim=-1)
            total_loss += loss.item()
            correct += int((predicted == batch.targets).sum())
            count += len(batch.targets)
            confusion += torch.bincount(
                batch.targets * model.num_classes + predicted,
                minlength=model.num_classes ** 2,
            ).reshape(model.num_classes, model.num_classes)
    return {"loss": total_loss / count, "accuracy": correct / count, "num_events": count,
            "confusion_matrix": confusion.tolist()}


def _diagnostics(
    events: Sequence[EventBatch], model: EventClassificationModel, batch_size: int,
) -> dict[str, Any]:
    batch = collate_event_batches(events[:batch_size])
    model.eval()
    with torch.no_grad():
        tokens = model.encoder.tokenizer(batch)
        embeddings = model.encoder.encoder(tokens)
        logits = model.head(embeddings)
    collections = {}
    for name, group in batch.objects.items():
        metadata = [event.objects[name].metadata for event in events]
        collections[name] = {
            "feature_names": group.metadata["feature_names"],
            "features_shape": list(group.features.shape), "mask_shape": list(group.mask.shape),
            "mask_dtype": str(group.mask.dtype), "valid_counts": group.mask.sum(1).tolist(),
            "padding_count": int((~group.mask).sum()),
            "original_objects_total": sum(entry["original_count"] for entry in metadata),
            "retained_objects_total": sum(entry["retained_count"] for entry in metadata),
            "truncated_objects_total": sum(entry["truncated_count"] for entry in metadata),
            "nonfinite_values_replaced": sum(entry["nonfinite_values"] for entry in metadata),
            "missing_optional_fields": sorted({field for entry in metadata for field in entry["missing_optional_fields"]}),
        }
    finite = all(torch.isfinite(group.features).all().item() for event in events for group in event.objects.values())
    return {
        "collections": collections, "tokens_shape": list(tokens.tokens.shape),
        "token_mask_shape": list(tokens.mask.shape), "group_ids_shape": list(tokens.group_ids.shape),
        "valid_tokens_per_event": tokens.mask.sum(1).tolist(),
        "embedding_shape": list(embeddings.shape), "logits_shape": list(logits.shape),
        "all_features_finite": finite,
        "tokens_embeddings_logits_finite": bool(all(torch.isfinite(tensor).all() for tensor in (tokens.tokens, embeddings, logits))),
        "shape_sample_event_indices": list(range(min(len(events), batch_size))),
    }


def train_foundation_smoke(
    processes: Mapping[str, str | Path | Sequence[str | Path]],
    *,
    view: str,
    model_config: Mapping[str, Any],
    max_events_per_class: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train all encoder/head parameters and save the best validation checkpoint.

    Inputs and converted events are held in RAM. This is deliberately a small
    CPU smoke run, without orchestration, dataset downloads or physics weights.
    """
    if len(processes) < 2:
        raise ValueError("Provide at least two distinct processes for classification.")
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("max_events_per_class", max_events_per_class)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if max_events_per_class < 3:
        raise ValueError("max_events_per_class must be at least 3.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    unknown = set(model_config) - {"model", "data"}
    if unknown:
        raise ValueError(f"Unknown foundation config sections: {sorted(unknown)}")
    cfg = dict(model_config.get("model", {}))
    if cfg.pop("name", "foundation_tiny") != "foundation_tiny":
        raise ValueError("Expected model.name: foundation_tiny.")
    perceiver_config = PerceiverConfig(**cfg)
    data_config = dict(model_config.get("data", {}))
    collection_limits = data_config.pop("collection_limits", None)
    specs = default_collection_specs(collection_limits)
    if unknown := set(data_config) - {"eta_clip", "value_clip"}:
        raise ValueError(f"Unknown foundation data settings: {sorted(unknown)}")
    # Resolve once for reproducible provenance and overlap checks.
    paths = {name: expand_parquet_paths(value) for name, value in processes.items()}
    events, label_map = load_process_events(paths, view, specs,
        max_events_per_class=max_events_per_class, **data_config)
    splits = stratified_event_split(events, seed)
    torch.manual_seed(seed)
    dims = collection_feature_dims(specs)
    model = EventClassificationModel(FoundationEncoder(dims, perceiver_config), len(label_map))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loaders = {
        name: DataLoader([events[index] for index in indices], batch_size=batch_size,
                         shuffle=name == "train", collate_fn=collate_event_batches,
                         generator=torch.Generator().manual_seed(seed))
        for name, indices in splits.items()
    }
    history = []
    best_loss, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, batch.targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training logits or loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise FloatingPointError("Non-finite model parameters after optimizer step.")
            loss_sum += loss.item() * len(batch.targets)
            count += len(batch.targets)
        validation = _evaluate(model, loaders["validation"])
        history.append({"epoch": epoch, "train_loss": loss_sum / count, "validation": validation})
        print(f"Epoch {epoch}/{epochs}: train_loss={loss_sum / count:.6f} "
              f"val_loss={validation['loss']:.6f} val_accuracy={validation['accuracy']:.4f}", flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    training_args = {"view": view, "max_events_per_class": max_events_per_class,
                     "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed}
    resolved_config = {
        "model": {"name": "foundation_tiny", **asdict(perceiver_config)},
        "data": {"collection_limits": dict(collection_limits or {}),
                 "eta_clip": data_config.get("eta_clip", 5.0), "value_clip": data_config.get("value_clip", 1.0e6)},
    }
    diagnostics = _diagnostics(events, model, batch_size)
    warnings = ["CPU, in-memory feasibility smoke; no physics benchmark or event weights.",
                "Events are the first capped rows of sorted files, then split by event; duplicated physics events are not detected."]
    for name, info in diagnostics["collections"].items():
        if info["missing_optional_fields"]:
            warnings.append(f"{name}: zero-filled optional fields {info['missing_optional_fields']}.")
        if info["nonfinite_values_replaced"]:
            warnings.append(f"{name}: replaced {info['nonfinite_values_replaced']} non-finite numeric entries with zero.")
    if sum(info["retained_objects_total"] for info in diagnostics["collections"].values()) == 0:
        warnings.append("All events are empty: predictions use only the learned latent prior.")
    metrics = {
        "view": view, "device": "cpu", "seed": seed, "label_map": label_map,
        "config": resolved_config, "training_args": training_args, "collection_feature_dims": dims,
        "input_files": {name: [str(path) for path in files] for name, files in paths.items()},
        "files_read": sorted({event.metadata["source_file"] for event in events}),
        "event_sources": [event.metadata for event in events],
        "split_indices": splits, "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "split_class_counts": {name: {process: sum(int(events[i].targets.item()) == label for i in indices)
                                      for process, label in label_map.items()} for name, indices in splits.items()},
        "history": history, "best_epoch": best_epoch, "checkpoint_selection": "minimum validation loss",
        "train": _evaluate(model, loaders["train"]),
        "validation": _evaluate(model, loaders["validation"]),
        "test": _evaluate(model, loaders["test"]), "diagnostics": diagnostics,
        "finite_gradients": True, "warnings": warnings,
    }
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state, "config": resolved_config,
        "collection_specs": [asdict(spec) for spec in specs], "collection_feature_dims": dims,
        "label_map": label_map, "training_args": training_args, "best_epoch": best_epoch,
        "split_indices": splits,
    }, output / "checkpoint.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metrics


__all__ = ["stratified_event_split", "train_foundation_smoke"]
