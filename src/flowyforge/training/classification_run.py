"""Single-device classification of bounded or full streaming COLLIDE samples."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from flowyforge.analysis.classification_report import (
    EvaluationScoreWriter, compute_binary_metrics, compute_confusion_metrics,
    read_evaluation_scores, write_classification_report,
)
from flowyforge.data_plugins.collide_v2.provenance import sha256_file
from flowyforge.core.config import load_yaml
from flowyforge.core.typing import EventBatch, ObjectGroup
from flowyforge.data_plugins.collide_v2.object_dataset import (
    collate_event_batches, collection_feature_dims, default_collection_specs,
)
from flowyforge.data_plugins.collide_v2.streaming import (
    StreamingDataset, bind_partition_manifest, build_manifest, inspect_dataset, load_partition_manifest,
)
from flowyforge.model_plugins import EventClassificationModel, FoundationEncoder, PerceiverConfig


def _known(mapping: Mapping[str, Any], allowed: set[str], section: str) -> None:
    if unknown := set(mapping) - allowed:
        raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")


def _positive_integer(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def load_run_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve YAML and explicit CLI overrides; reject misspelled settings."""
    path = Path(path).expanduser().resolve()
    raw = load_yaml(path)
    return _resolve_run_config(raw, path, overrides)


def _resolve_run_config(raw: Mapping[str, Any], path: Path,
                        overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Shared reader/model/optimizer configuration, also used by pretraining."""
    overrides = dict(overrides or {})
    _known(raw, {"mode", "model_config", "output_dir", "seed", "sources", "data", "training", "partition_manifest", "evaluation"}, "run")
    _known(overrides, {"mode", "processes", "data_root", "process_names", "max_events_per_class",
                       "view", "device", "epochs", "batch_size", "num_workers", "seed",
                       "output_dir", "model_config", "partition_manifest"}, "override")
    mode = overrides.get("mode", raw.get("mode", "small"))
    if mode not in {"small", "eos"}:
        raise ValueError("mode must be small or eos.")
    sources = raw.get("sources", {})
    _known(sources, {"small", "eos"}, "sources")
    source = dict(sources.get(mode, {}))
    _known(source, {"processes", "root", "process_names", "max_events_per_class"}, "source")
    source.setdefault("max_events_per_class", 128 if mode == "small" else None)
    for key, destination in (("processes", "processes"), ("data_root", "root"),
                             ("process_names", "process_names"), ("max_events_per_class", "max_events_per_class")):
        if key in overrides:
            source[destination] = overrides[key]
    if type(source["max_events_per_class"]) is int and source["max_events_per_class"] == 0:
        source["max_events_per_class"] = None
    if source["max_events_per_class"] is not None:
        _positive_integer(source["max_events_per_class"], "max_events_per_class", 3)
    if mode == "small" and not source.get("processes"):
        raise ValueError("small mode needs at least two --process NAME=PATH entries or sources.small.processes in YAML.")
    if mode == "eos":
        source.setdefault("root", "/eos/project/f/foundational-model-dataset/samples/")

    model_path = Path(overrides.get("model_config", raw.get("model_config", "../models/foundation_tiny.yaml"))).expanduser()
    if not model_path.is_absolute():
        model_path = (Path.cwd() if "model_config" in overrides else path.parent) / model_path
    model_config = load_yaml(model_path)
    _known(model_config, {"model", "data"}, "model config")
    model = dict(model_config.get("model", {}))
    if model.pop("name", "foundation_tiny") != "foundation_tiny":
        raise ValueError("Only model.name: foundation_tiny is supported by this runner.")
    model = {"name": "foundation_tiny", **asdict(PerceiverConfig(**model))}
    data = {"view": "FullReco", "validation_fraction": 0.2, "test_fraction": 0.2,
            "read_batch_size": 128, "shuffle_buffer": 512, "audit_events_per_class": 32,
            "units": {"momentum": "GeV", "angle": "rad"}, "eta_clip": 5.0, "value_clip": 1.0e6,
            **model_config.get("data", {}), **raw.get("data", {})}
    if "view" in overrides:
        data["view"] = overrides["view"]
    _known(data, {"view", "validation_fraction", "test_fraction", "read_batch_size", "shuffle_buffer",
                  "audit_events_per_class", "units", "collection_limits", "eta_clip", "value_clip"}, "data")
    if data["view"] not in {"FullReco", "L1T"}:
        raise ValueError("view must be FullReco or L1T.")
    fractions = [data["validation_fraction"], data["test_fraction"]]
    if any(not math.isfinite(x) or x <= 0 for x in fractions) or sum(fractions) >= 1:
        raise ValueError("validation_fraction and test_fraction must be positive and sum to < 1.")
    for key in ("read_batch_size", "shuffle_buffer", "audit_events_per_class"):
        _positive_integer(data[key], key)
    for key in ("eta_clip", "value_clip"):
        if not math.isfinite(data[key]) or data[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    _known(data["units"], {"momentum", "angle"}, "units")
    if data["units"].get("momentum") not in {"GeV", "MeV"} or data["units"].get("angle") not in {"rad", "deg"}:
        raise ValueError("Input units must specify momentum: GeV|MeV and angle: rad|deg.")
    specs = default_collection_specs(data.get("collection_limits"))
    if any(spec.max_objects is None for spec in specs):
        raise ValueError("Set a finite collection_limits entry for every enabled collection in streaming runs.")

    training = {"epochs": 10, "batch_size": 32, "learning_rate": 1.0e-3, "weight_decay": 1.0e-4,
                "gradient_clip_norm": 1.0, "device": "auto", "num_workers": 0,
                "cpu_threads": 4, "log_every_batches": 100, **raw.get("training", {})}
    _known(training, {"epochs", "batch_size", "learning_rate", "weight_decay", "gradient_clip_norm",
                      "device", "num_workers", "cpu_threads", "log_every_batches"}, "training")
    for key in ("epochs", "batch_size", "device", "num_workers"):
        if key in overrides:
            training[key] = overrides[key]
    for key in ("epochs", "batch_size", "cpu_threads", "log_every_batches"):
        _positive_integer(training[key], key)
    _positive_integer(training["num_workers"], "num_workers", 0)
    for key in ("learning_rate", "gradient_clip_norm", "weight_decay"):
        value = training[key]
        if not math.isfinite(value) or value < 0 or (key != "weight_decay" and value == 0):
            raise ValueError(f"Invalid {key}: {value}")
    seed = overrides.get("seed", raw.get("seed", 42))
    _positive_integer(seed, "seed", 0)
    partition_path = overrides.get("partition_manifest", raw.get("partition_manifest"))
    if partition_path is not None:
        if not isinstance(partition_path, (str, Path)) or not str(partition_path).strip():
            raise ValueError("partition_manifest must be a nonempty path or null.")
        partition_path = str(Path(partition_path).expanduser())
    evaluation = {"signal_process": None, "background_acceptance": 0.01,
                  "min_background_events": 10, "score_shard_rows": 10000, **raw.get("evaluation", {})}
    _known(evaluation, {"signal_process", "background_acceptance", "min_background_events", "score_shard_rows"}, "evaluation")
    signal = evaluation["signal_process"]
    if signal is not None and (not isinstance(signal, str) or not signal.strip()):
        raise ValueError("evaluation.signal_process must be a nonempty class name or null.")
    fraction = evaluation["background_acceptance"]
    if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("evaluation.background_acceptance must be in (0, 1].")
    for key in ("min_background_events", "score_shard_rows"):
        _positive_integer(evaluation[key], "evaluation." + key)
    if signal is not None and partition_path is None:
        raise ValueError("Binary evaluation requires partition_manifest for stable event identities.")
    return {"mode": mode, "source": source, "model": model, "data": data, "training": training,
            "partition_manifest": partition_path, "evaluation": evaluation,
            "seed": seed, "output_dir": str(Path(overrides.get("output_dir", raw.get("output_dir", "outputs/classification"))).expanduser()),
            "config_path": str(path), "model_config_path": str(model_path.resolve())}


def resolve_device(requested: str) -> torch.device:
    """Auto chooses CUDA, then MPS, then CPU; explicit unavailable devices fail."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda[:index], or mps.")
    if device.type == "cuda" and (not torch.cuda.is_available() or (
        device.index is not None and device.index >= torch.cuda.device_count()
    )):
        raise ValueError(f"Requested device {device} is unavailable; use --device cpu or a GPU node.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS device is unavailable.")
    return device


def batch_to_device(batch: EventBatch, device: torch.device) -> EventBatch:
    """Move features, masks and labels together, preserving CPU provenance."""
    return EventBatch(
        objects={name: ObjectGroup(name, group.features.to(device),
                                   None if group.mask is None else group.mask.to(device), group.metadata,
                                   None if group.feature_valid is None else group.feature_valid.to(device),
                                   None if group.feature_mask is None else group.feature_mask.to(device))
                 for name, group in batch.objects.items()},
        targets=None if batch.targets is None else batch.targets.to(device), metadata=batch.metadata,
    )


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest_digest(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _make_model(config: dict[str, Any], label_map: dict[str, int]) -> EventClassificationModel:
    parameters = {key: value for key, value in config["model"].items() if key != "name"}
    dims = collection_feature_dims(default_collection_specs(config["data"]["collection_limits"]))
    return EventClassificationModel(FoundationEncoder(dims, PerceiverConfig(**parameters)), len(label_map))


def _update_diagnostics(diagnostics: dict[str, Any], batch: EventBatch) -> None:
    diagnostics["num_events"] = diagnostics.get("num_events", 0) + len(batch.targets)
    collections = diagnostics.setdefault("collections", {})
    for name, group in batch.objects.items():
        totals = collections.setdefault(name, {
            "feature_names": group.metadata["feature_names"], "original_objects_total": 0,
            "retained_objects_total": 0, "truncated_objects_total": 0, "truncated_events": 0,
            "nonfinite_values_replaced": 0, "missing_optional_fields": [], "missing_optional_events": 0,
        })
        for event in group.metadata["events"]:
            for source, destination in (("original_count", "original_objects_total"),
                                        ("retained_count", "retained_objects_total"),
                                        ("truncated_count", "truncated_objects_total"),
                                        ("nonfinite_values", "nonfinite_values_replaced")):
                totals[destination] += event[source]
            totals["truncated_events"] += int(event["truncated_count"] > 0)
            totals["missing_optional_events"] += int(bool(event["missing_optional_fields"]))
            totals["missing_optional_fields"] = sorted(set(totals["missing_optional_fields"]) | set(event["missing_optional_fields"]))


def _run_epoch(model: EventClassificationModel, loader: DataLoader, device: torch.device,
               label_map: dict[str, int], *, optimizer: torch.optim.Optimizer | None = None,
               gradient_clip_norm: float = 1.0, log_every_batches: int = 100,
               description: str = "evaluation", diagnostics: dict[str, Any] | None = None,
               score_writer: EvaluationScoreWriter | None = None) -> dict[str, Any]:
    training = optimizer is not None
    if training and score_writer is not None:
        raise ValueError("Persisted evaluation scores must use a frozen selected checkpoint, not training batches.")
    model.train(training)
    confusion = torch.zeros((len(label_map), len(label_map)), dtype=torch.long, device=device)
    loss_sum, count = 0.0, 0
    started = time.monotonic()
    with torch.set_grad_enabled(training):
        for step, cpu_batch in enumerate(loader, 1):
            if diagnostics is not None:
                _update_diagnostics(diagnostics, cpu_batch)
            batch = batch_to_device(cpu_batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = nn.functional.cross_entropy(logits, batch.targets)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite logits/loss during {description}.")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
            predicted = logits.detach().argmax(dim=-1)
            if score_writer is not None:
                signal_label = score_writer.context["signal_label"]
                # Float64 on CPU also works with MPS (which lacks float64 tensors).
                scores = logits.detach().cpu().double().softmax(dim=-1)[:, signal_label]
                score_writer.append(cpu_batch.metadata["events"], cpu_batch.targets.numpy(), scores.cpu().numpy())
            # bincount is accumulated on CPU for MPS, whose support is limited.
            flat = batch.targets * len(label_map) + predicted
            if device.type == "mps":
                flat = flat.cpu()
            confusion += torch.bincount(flat, minlength=len(label_map) ** 2).reshape(confusion.shape).to(device)
            n = len(batch.targets)
            loss_sum += loss.detach().item() * n
            count += n
            if step % log_every_batches == 0:
                print(f"{description}: {count:,} events, loss={loss_sum/count:.5f}, "
                      f"{count / max(time.monotonic() - started, 1e-9):.1f} events/s", flush=True)
    if not count:
        raise ValueError(f"Empty {description} split. Increase the sample or adjust split fractions.")
    result = compute_confusion_metrics(confusion.cpu().tolist(), label_map)
    result.update(loss=loss_sum / count, duration_seconds=time.monotonic() - started)
    return result


def _training_protocol(config):
    """Comparable scientific settings; runtime placement/logging can differ."""
    return {"seed": config["seed"], "model": config["model"],
            "data": {key: value for key, value in config["data"].items() if key != "view"},
            "training": {key: value for key, value in config["training"].items()
                         if key not in {"device", "cpu_threads", "log_every_batches"}},
            "collection_feature_dims": collection_feature_dims(default_collection_specs(config["data"]["collection_limits"]))}


def _initial_state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _binary_settings(config, manifest):
    settings = config.get("evaluation", {})
    signal = settings.get("signal_process")
    if signal is None:
        return None
    if len(manifest["label_map"]) != 2 or signal not in manifest["label_map"]:
        raise ValueError("Binary evaluation requires exactly two classes and an explicit signal_process in label_map.")
    if manifest["version"] != 2:
        raise ValueError("Binary evaluation requires shared partitions (dataset manifest v2).")
    return settings


def _evaluate_split(model, loader, device, manifest, config, split, output, metrics):
    settings = _binary_settings(config, manifest)
    writer = None
    if settings:
        signal = settings["signal_process"]
        context = {key: metrics[key] for key in ("partition_sha256", "dataset_manifest_sha256", "checkpoint_sha256", "view")}
        context.update(signal_process=signal, signal_label=manifest["label_map"][signal], label_map=manifest["label_map"])
        writer = EvaluationScoreWriter(output, split, context=context, shard_rows=settings["score_shard_rows"])
    result = _run_epoch(model, loader, device, manifest["label_map"],
                        log_every_batches=config["training"]["log_every_batches"], description=f"best model {split}",
                        diagnostics=metrics["diagnostics"], score_writer=writer)
    if writer:
        result["score_manifest"] = writer.finish()
        _, ids, labels, scores = read_evaluation_scores(Path(output) / result["score_manifest"])
        expected = {}
        for process in manifest["processes"]:
            keys = [key for info in process["files"] for key in info["event_ids"]]
            expected.update((keys[index], process["label"]) for index in process["split_indices"][split])
        if dict(zip(ids, labels.tolist())) != expected:
            raise ValueError(f"Evaluation score identities/labels differ from the {split} partition.")
        binary = compute_binary_metrics(scores, labels, signal_label=writer.context["signal_label"],
                    background_acceptance=settings["background_acceptance"], min_background_events=settings["min_background_events"])
        binary.update(signal_process=settings["signal_process"],
                      background_process=next(name for name in manifest["label_map"] if name != settings["signal_process"]))
        result["binary"] = binary
    return result


def _reader_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {key: config["data"][key] for key in
            ("view", "read_batch_size", "units", "collection_limits", "eta_clip", "value_clip")}


def _init_worker(worker_id: int) -> None:
    """Keep worker IPC descriptors out of exec'ed shared-memory helpers."""
    if os.name == "posix":
        # Spawn passes tracker and worker-sentinel pipes as inheritable FDs.
        # torch_shm_manager must not keep these alive after the worker exits:
        # that delays worker joins and can deadlock Python's finalization.
        # CLOEXEC preserves the worker's open FDs, closing only copies in a
        # subsequently exec'ed helper. Leave standard streams available.
        try:
            descriptors = [int(name) for name in os.listdir("/dev/fd")]
        except FileNotFoundError:
            descriptors = range(3, os.sysconf("SC_OPEN_MAX"))
        for fd in descriptors:
            if fd >= 3:
                try:
                    os.set_inheritable(fd, False)
                except OSError as exc:
                    if exc.errno != errno.EBADF:  # e.g. the closed /dev/fd iterator
                        raise


def _loader(manifest: dict[str, Any], config: dict[str, Any], split: str,
            *, purpose: str = "downstream") -> DataLoader:
    dataset = StreamingDataset(manifest, split, shuffle_buffer=config["data"]["shuffle_buffer"],
                               purpose=purpose, **_reader_settings(config))
    workers = config["training"]["num_workers"]
    # Spawn avoids inheriting a CUDA context. Each epoch gets the updated epoch
    # seed; worker sharding belongs to the iterable dataset, not a sampler.
    options = {"multiprocessing_context": "spawn", "prefetch_factor": 2,
               "worker_init_fn": _init_worker} if workers else {}
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], num_workers=workers,
                      collate_fn=collate_event_batches, generator=torch.Generator().manual_seed(config["seed"]), **options)


def _base_metrics(config: dict[str, Any], manifest: dict[str, Any], preflight: dict[str, Any],
                  device: torch.device) -> dict[str, Any]:
    return {
        "config": config, "mode": config["mode"], "view": config["data"]["view"],
        "device": str(device), "seed": config["seed"], "label_map": manifest["label_map"],
        "checkpoint_selection": "minimum validation loss", "dataset_manifest_sha256": _manifest_digest(manifest),
        "partition_sha256": manifest.get("partition_sha256"),
        "expected_split_class_counts": {
            split: {process["name"]: process["split_counts"][split] for process in manifest["processes"]}
            for split in ("train", "validation", "test")
        },
        "preflight": preflight,
        "warnings": list(preflight.get("warnings", [])) + [
            "Preliminary, unweighted event classification; no luminosity or cross-section weights.",
            "Splits separate source rows. Duplicate physics events in distinct files are not detected.",
        ],
    }


def _record_split_counts(metrics: dict[str, Any], train: dict[str, Any] | None = None) -> None:
    splits = {name: metrics[name] for name in ("validation", "test")}
    if train is not None:
        splits["train"] = train
    metrics["split_sizes"] = {name: result["num_events"] for name, result in splits.items()}
    metrics["split_class_counts"] = {
        name: {process: int(sum(result["confusion_matrix"][label]))
               for process, label in metrics["label_map"].items()}
        for name, result in splits.items()
    }
    for split, counts in metrics["split_class_counts"].items():
        if counts != metrics["expected_split_class_counts"][split]:
            raise RuntimeError(f"Incomplete or duplicated {split} stream: counts differ from the frozen manifest.")


def _finish_diagnostics(diagnostics: dict[str, Any]) -> None:
    events = diagnostics.get("num_events", 0)
    for totals in diagnostics.get("collections", {}).values():
        original = totals["original_objects_total"]
        totals["truncated_object_fraction"] = totals["truncated_objects_total"] / original if original else 0.0
        totals["truncated_event_fraction"] = totals["truncated_events"] / events if events else 0.0
        totals["missing_optional_event_fraction"] = totals["missing_optional_events"] / events if events else 0.0


def run_classification(config: dict[str, Any], *, inspect_only: bool = False) -> dict[str, Any]:
    """Preflight all file schemas, optionally train and evaluate one selected model.

    Checkpoints are inference artifacts. A completed or interrupted training
    directory is protected from accidental reuse; choose a new output directory.
    """
    output = Path(config["output_dir"])
    if any((output / name).exists() for name in ("checkpoint.pt", "history.json", "metrics.json", "scores")):
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}. Choose a new --output-dir.")
    device = resolve_device(config["training"]["device"]) if not inspect_only else torch.device("cpu")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print("Building dataset manifest and checking Parquet schemas...", flush=True)
    reader = {key: value for key, value in _reader_settings(config).items() if key != "read_batch_size"}
    if config.get("partition_manifest"):
        partitions = load_partition_manifest(config["partition_manifest"])
        for key in ("validation_fraction", "test_fraction"):
            if config["data"][key] != partitions[key]:
                raise ValueError(f"{key} differs from the saved partition manifest.")
        manifest = bind_partition_manifest(partitions, config["source"], **reader)
        _write_json(output / "partitions.json", partitions)
    else:
        manifest = build_manifest(config["source"], seed=config["seed"],
                                  validation_fraction=config["data"]["validation_fraction"],
                                  test_fraction=config["data"]["test_fraction"], **reader)
    _binary_settings(config, manifest)
    preflight = inspect_dataset(manifest, audit_events_per_class=config["data"]["audit_events_per_class"],
                                **_reader_settings(config))
    _write_json(output / "dataset_manifest.json", manifest)
    _write_json(output / "preflight.json", preflight)
    print(f"Classes: {manifest['label_map']}; device: {device}", flush=True)
    for warning in preflight.get("warnings", []):
        print(f"Preflight: {warning}", flush=True)
    if inspect_only:
        return preflight

    seed, settings = config["seed"], config["training"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.set_num_threads(settings["cpu_threads"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = _make_model(config, manifest["label_map"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    loaders = {name: _loader(manifest, config, name) for name in ("train", "validation", "test")}
    history, best_loss, best_epoch = [], math.inf, 0
    diagnostics: dict[str, Any] = {"scope": "all selected rows once: first train epoch, final validation and test"}
    first_train = None
    metrics = _base_metrics(config, manifest, preflight, device)
    metrics.update(initialization="random_from_scratch", initial_state_sha256=_initial_state_digest(model),
                   training_protocol=_training_protocol(config))
    for epoch in range(1, settings["epochs"] + 1):
        started = time.monotonic()
        loaders["train"].dataset.set_epoch(epoch)
        train = _run_epoch(model, loaders["train"], device, manifest["label_map"], optimizer=optimizer,
                           gradient_clip_norm=settings["gradient_clip_norm"], log_every_batches=settings["log_every_batches"],
                           description=f"epoch {epoch} train", diagnostics=diagnostics if epoch == 1 else None)
        if first_train is None:
            first_train = train
        validation = _run_epoch(model, loaders["validation"], device, manifest["label_map"],
                                log_every_batches=settings["log_every_batches"], description=f"epoch {epoch} validation")
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                        "train_num_events": train["num_events"], "validation": validation,
                        "duration_seconds": time.monotonic() - started})
        _write_json(output / "history.json", history)
        print(f"Epoch {epoch}/{settings['epochs']}: train loss={train['loss']:.5f}, "
              f"validation loss={validation['loss']:.5f}, accuracy={validation['accuracy']:.4f}, "
              f"macro F1={validation['macro_f1']:.4f}", flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            temporary = output / "checkpoint.pt.tmp"
            torch.save({"format_version": 1, "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "config": config, "label_map": manifest["label_map"], "best_epoch": best_epoch,
                        **{key: metrics[key] for key in ("initialization", "initial_state_sha256", "training_protocol")},
                        "dataset_manifest_sha256": metrics["dataset_manifest_sha256"],
                        "collection_feature_dims": collection_feature_dims(default_collection_specs(config["data"]["collection_limits"]))}, temporary)
            temporary.replace(output / "checkpoint.pt")

    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics.update(history=history, best_epoch=best_epoch, diagnostics=diagnostics)
    metrics["checkpoint_sha256"] = sha256_file(output / "checkpoint.pt")
    for split in ("validation", "test"):
        metrics[split] = _evaluate_split(model, loaders[split], device, manifest, config, split, output, metrics)
    _record_split_counts(metrics, first_train)
    _finish_diagnostics(diagnostics)
    _write_json(output / "metrics.json", metrics)
    write_classification_report(metrics, output)
    return metrics


def evaluate_run(checkpoint_path: str | Path, *, output_dir: str | Path,
                  device: str | None = None, batch_size: int | None = None,
                  num_workers: int | None = None, data_root: str | Path | None = None) -> dict[str, Any]:
    """Evaluate the checkpoint on its frozen validation/test rows, never resplit."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    output = Path(output_dir).expanduser()
    if any((output / name).exists() for name in ("checkpoint.pt", "history.json", "scores")):
        raise FileExistsError(f"Evaluation output contains an existing training run: {output}. Choose a separate directory.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Expected a streaming classification checkpoint (format_version=1).")
    manifest = json.loads((checkpoint_path.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
    if _manifest_digest(manifest) != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the manifest saved with the checkpoint.")
    config = checkpoint["config"]
    config["output_dir"] = str(output_dir)
    if manifest.get("version") == 2:
        partitions = load_partition_manifest(checkpoint_path.parent / "partitions.json")
        if partitions["sha256"] != manifest["partition_sha256"]:
            raise ValueError("Partition manifest differs from the checkpoint's saved memberships.")
        source = config["source"] if data_root is None else {
            "root": str(data_root), "process_names": list(manifest["label_map"]), "max_events_per_class": None}
        config["source"] = source
        manifest = bind_partition_manifest(partitions, source, **{
            key: value for key, value in _reader_settings(config).items() if key != "read_batch_size"})
    elif data_root is not None:
        raise ValueError("Relocation requires a run with shared partitions (dataset manifest v2); legacy v1 paths are frozen.")
    for key, value in (("device", device), ("batch_size", batch_size), ("num_workers", num_workers)):
        if value is not None:
            config["training"][key] = value
    _positive_integer(config["training"]["batch_size"], "batch_size")
    _positive_integer(config["training"]["num_workers"], "num_workers", 0)
    resolved_device = resolve_device(config["training"]["device"])
    torch.set_num_threads(config["training"]["cpu_threads"])
    model = _make_model(config, manifest["label_map"]).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    preflight = json.loads((checkpoint_path.parent / "preflight.json").read_text(encoding="utf-8"))
    metrics = _base_metrics(config, manifest, preflight, resolved_device)
    metrics.update({key: checkpoint.get(key) for key in ("initialization", "initial_state_sha256", "training_protocol")})
    metrics["checkpoint_sha256"] = sha256_file(checkpoint_path)
    history_path = checkpoint_path.parent / "history.json"
    metrics.update(best_epoch=checkpoint["best_epoch"], history=json.loads(history_path.read_text()) if history_path.exists() else [],
                   checkpoint_path=str(checkpoint_path.resolve()), diagnostics={"scope": "validation and test rows only"})
    for split in ("validation", "test"):
        metrics[split] = _evaluate_split(model, _loader(manifest, config, split), resolved_device,
                                        manifest, config, split, output, metrics)
    _record_split_counts(metrics)
    _finish_diagnostics(metrics["diagnostics"])
    output.mkdir(parents=True, exist_ok=True)
    if manifest.get("version") == 2:
        _write_json(output / "dataset_manifest.json", manifest)
        _write_json(output / "partitions.json", partitions)
    _write_json(output / "metrics.json", metrics)
    write_classification_report(metrics, output)
    return metrics


__all__ = ["load_run_config", "resolve_device", "batch_to_device", "run_classification", "evaluate_run"]
