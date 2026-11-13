"""Single-device classification of bounded or full streaming COLLIDE samples."""

from __future__ import annotations

import errno
import copy
import hashlib
import json
import math
import os
import random
import pickle
import platform
import shutil
import tempfile
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
from flowyforge.model_plugins.foundation_encoder import encoder_contract
from flowyforge.training.checkpointing import _rng_state, _restore_rng, _save_checkpoint, state_digest


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
    _known(raw, {"mode", "model_config", "output_dir", "seed", "sources", "data", "training", "partition_manifest", "evaluation", "downstream"}, "run")
    _known(overrides, {"mode", "processes", "data_root", "process_names", "max_events_per_class",
                       "view", "device", "epochs", "batch_size", "num_workers", "seed",
                       "output_dir", "model_config", "partition_manifest", "downstream_mode", "encoder_checkpoint"}, "override")
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
    downstream = None
    if "downstream" in raw or any(k in overrides for k in ("downstream_mode", "encoder_checkpoint")):
        downstream = {"mode": "scratch", "encoder_checkpoint": None, **raw.get("downstream", {})}
        _known(downstream, {"mode", "encoder_checkpoint"}, "downstream")
        for source_key, key in (("downstream_mode", "mode"), ("encoder_checkpoint", "encoder_checkpoint")):
            if source_key in overrides:
                downstream[key] = overrides[source_key]
        if downstream["mode"] not in {"scratch", "linear_probe", "finetune"}:
            raise ValueError("downstream.mode must be scratch, linear_probe or finetune.")
        checkpoint = downstream["encoder_checkpoint"]
        if downstream["mode"] == "scratch":
            if checkpoint is not None:
                raise ValueError("scratch must not specify an encoder_checkpoint.")
        elif not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
            raise ValueError("linear_probe/finetune requires encoder_checkpoint (an exported encoder.pt).")
        else:
            downstream["encoder_checkpoint"] = str(Path(checkpoint).expanduser())
        if partition_path is None or signal is None:
            raise ValueError("Downstream benchmarking requires shared partition_manifest and evaluation.signal_process.")
    return {"mode": mode, "source": source, "model": model, "data": data, "training": training,
            **({"downstream": downstream} if downstream is not None else {}),
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


def _make_model(config: dict[str, Any], label_map: dict[str, int], *, initialize_encoder=True) -> EventClassificationModel:
    parameters = {key: value for key, value in config["model"].items() if key != "name"}
    dims = collection_feature_dims(default_collection_specs(config["data"]["collection_limits"]))
    downstream = config.get("downstream")
    if downstream is None:
        return EventClassificationModel(FoundationEncoder(dims, PerceiverConfig(**parameters)), len(label_map))
    if downstream["mode"] not in {"scratch", "linear_probe", "finetune"}:
        raise ValueError("Unsupported downstream mode in model configuration.")
    frozen = downstream["mode"] == "linear_probe"
    if initialize_encoder and downstream["mode"] != "scratch":
        return EventClassificationModel.from_encoder_checkpoint(
            downstream["encoder_checkpoint"], len(label_map), expected_contract=encoder_contract(config["model"], config["data"]),
            head_type="linear", freeze_encoder=frozen)
    # Evaluation restores the complete classifier: the original encoder.pt is not needed.
    return EventClassificationModel(FoundationEncoder(dims, PerceiverConfig(**parameters), mask_indicators=True),
                                    len(label_map), head_type="linear", freeze_encoder=frozen)


def _downstream_metadata(model, config, manifest):
    settings = config["downstream"]
    source = None
    if settings["mode"] != "scratch":
        provenance = model.encoder.checkpoint_provenance
        if (provenance.get("partition_sha256") != manifest.get("partition_sha256")
            or provenance.get("pretraining_split") != "train" or provenance.get("checkpoint_selection_split") != "validation"
            or provenance.get("test_used") is not False):
            raise ValueError("Encoder provenance must match downstream partitions, train-only pretraining and validation selection; test must be excluded.")
        source = {"path": str(Path(settings["encoder_checkpoint"]).resolve()),
                  "sha256": model.encoder.checkpoint_sha256, "provenance": provenance}
    return {"version": 1, "mode": settings["mode"], "head_type": "linear",
            "freezing": "encoder_frozen_eval" if model.freeze_encoder else "none",
            "encoder_source": source, "initial_encoder_sha256": _initial_state_digest(model.encoder),
            "initial_head_sha256": _initial_state_digest(model.head),
            "parameters": _parameter_counts(model)}


def _parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "encoder_trainable": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
            "head_trainable": sum(p.numel() for p in model.head.parameters() if p.requires_grad)}


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
    result["events_per_second"] = count / max(result["duration_seconds"], 1e-9)
    return result


def _performance(history):
    duration = sum(item.get("train_duration_seconds", 0) for item in history)
    events = sum(item["train_num_events"] for item in history)
    return {"train_duration_seconds": duration, "train_events": events,
            "train_events_per_second": events / duration if duration else None,
            "selection_validation_seconds": sum(item["validation"]["duration_seconds"] for item in history),
            "epoch_duration_seconds": sum(item["duration_seconds"] for item in history),
            "scope": "epoch train timing includes I/O and optimization; excludes preflight, final evaluation and reports"}


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


def _resume_protocol(config):
    protocol = copy.deepcopy(config)
    for key in ("config_path", "model_config_path", "output_dir"):
        protocol.pop(key, None)
    protocol["training"].pop("epochs")
    return protocol


def _classification_contract(config):
    contract = encoder_contract(config["model"], config["data"])
    contract["mask_indicators"] = bool(config.get("downstream"))
    return contract


def _load_training_state(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("expected a checkpoint mapping")
        return payload
    except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Unreadable/corrupt classification checkpoint {path}: {exc}") from exc


def _read_resume(resume, output, config):
    requested = Path(resume).expanduser().resolve()
    # Accept the older example spelling, but NEVER treat inference weights as resume state.
    path = requested.with_name("last_checkpoint.pt") if requested.name == "checkpoint.pt" else requested
    if path != (output / "last_checkpoint.pt").resolve():
        raise ValueError("Resume requires last_checkpoint.pt in the same --output-dir.")
    if not path.is_file():
        raise ValueError("No last_checkpoint.pt: old inference checkpoints lack optimizer/RNG state and cannot resume training.")
    fingerprint = sha256_file(path)
    saved = _load_training_state(path)
    if not isinstance(saved, dict) or saved.get("kind") != "flowyforge.classification_resume" or saved.get("version") != 1:
        raise ValueError("Expected classification last_checkpoint.pt v1 with optimizer/RNG; inference checkpoints cannot resume training.")
    required = {"epoch", "config", "protocol", "model_state_dict", "optimizer_state_dict", "optimizer_parameters",
                "rng", "best_checkpoint", "best_epoch", "best_loss", "history", "initial_metadata", "first_train",
                "diagnostics", "dataset_manifest_sha256", "partition_sha256", "source_sha256", "input_contract",
                "training_run", "runtime", "state_sha256"}
    if required - saved.keys():
        raise ValueError(f"Incomplete resume checkpoint: missing {sorted(required - saved.keys())}.")
    try:
        if saved["state_sha256"] != state_digest({k: v for k, v in saved.items() if k != "state_sha256"}):
            raise ValueError("Resume checkpoint integrity checksum mismatch.")
        if saved["protocol"] != _resume_protocol(saved["config"]) or saved["protocol"] != _resume_protocol(config):
            raise ValueError("Incompatible resume configuration/protocol: only training.epochs may increase.")
        epoch, history, best = saved["epoch"], saved["history"], saved["best_checkpoint"]
        if (type(epoch) is not int or epoch < 1 or len(history) != epoch
                or [h["epoch"] for h in history] != list(range(1, epoch + 1))
                or any(not math.isfinite(h["validation"]["loss"]) for h in history)):
            raise ValueError("Invalid/incomplete resume history or completed epoch.")
        chosen = min(history, key=lambda h: h["validation"]["loss"])
        if (saved["best_epoch"] != chosen["epoch"] or saved["best_loss"] != chosen["validation"]["loss"]
                or best["best_epoch"] != saved["best_epoch"] or best["best_validation_loss"] != saved["best_loss"]
                or best["config"] != saved["config"] or best["training_protocol"] != _training_protocol(saved["config"])
                or best["format_version"] != (2 if config.get("downstream") else 1)):
            raise ValueError("Resume best checkpoint is inconsistent with validation history/configuration.")
        if config["training"]["epochs"] < max(epoch, saved["config"]["training"]["epochs"]):
            raise ValueError("--epochs is the TOTAL horizon; it cannot shorten the saved training horizon.")
        for key, value in saved["initial_metadata"].items():
            if best.get(key) != value:
                raise ValueError(f"Resume initialization/provenance mismatch: {key}.")
        if saved["input_contract"] != _classification_contract(config):
            raise ValueError("Incompatible classification resume architecture/input contract.")
        if sha256_file(path) != fingerprint:
            raise ValueError("Resume checkpoint changed while loading.")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError(f"Malformed/incomplete resume checkpoint: {exc}") from exc
    return saved, {"path": str(path), "sha256": fingerprint, "saved_epoch": saved["epoch"]}


def _runtime(device):
    return {"device": str(device), "torch": str(torch.__version__), "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0}


def _training_run(saved, source, start, epoch, config, runtime):
    sessions = copy.deepcopy(saved["training_run"]["sessions"]) if saved else []
    sessions.append({"start_epoch": start + 1, "end_epoch": epoch, "epochs_executed": epoch - start,
                     "requested_epochs": config["training"]["epochs"], "resume_source": source})
    return {"resumed": saved is not None, "start_epoch": start + 1, "end_epoch": epoch,
            "epochs_completed": epoch, "epochs_this_invocation": epoch - start, "resume_source": source,
            "sessions": sessions, "runtime": runtime,
            "reproducibility": "Exact tensor/metric replay tested on CPU, num_workers=0, same software/hardware and config; "
                               "timing differs. Worker processes restart at epoch boundaries; no guarantee of bitwise "
                               "identity across accelerators, workers, library versions or hardware. No mid-epoch resume."}


def _validate_restored_training(model, optimizer, saved, config, manifest, device, loaders):
    """Validate all restored state before replacing any existing run artifacts."""
    try:
        names = [name for name, p in model.named_parameters() if p.requires_grad]
        if names != saved["optimizer_parameters"]:
            raise ValueError("Resume optimizer parameter names/freezing differ.")
        state = saved["optimizer_state_dict"]
        if len(state["param_groups"]) != 1 or not state["state"]:
            raise ValueError("Missing or invalid AdamW optimizer state.")
        expected = optimizer.state_dict()["param_groups"][0]
        if state["param_groups"][0] != expected:
            raise ValueError("Resume AdamW parameter groups/hyperparameters differ.")
        parameters = [p for p in model.parameters() if p.requires_grad]
        indices = state["param_groups"][0]["params"]
        if set(state["state"]) - set(indices):
            raise ValueError("Unknown optimizer parameter in resume state.")
        for index, value in state["state"].items():
            param = parameters[indices.index(index)]
            if (not {"step", "exp_avg", "exp_avg_sq"} <= value.keys() or value["step"].numel() != 1
                    or not torch.isfinite(value["step"]) or value["step"] < 1):
                raise ValueError("Incomplete/nonfinite AdamW state.")
            for key in ("exp_avg", "exp_avg_sq"):
                if value[key].shape != param.shape or not torch.isfinite(value[key]).all():
                    raise ValueError("Invalid AdamW moment shape/value.")
        best = saved["best_checkpoint"]
        model.load_state_dict(best["model_state_dict"], strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("Nonfinite best-model weights in resume checkpoint.")
        metadata = saved["initial_metadata"].get("downstream")
        if metadata:
            mode = config["downstream"]["mode"]
            if (metadata["mode"] != mode or metadata["head_type"] != "linear"
                    or metadata["freezing"] != ("encoder_frozen_eval" if mode == "linear_probe" else "none")
                    or metadata["parameters"] != _parameter_counts(model)
                    or metadata["selected_encoder_sha256"] != _initial_state_digest(model.encoder)
                    or metadata["selected_head_sha256"] != _initial_state_digest(model.head)):
                raise ValueError("Resume downstream best weights/freezing metadata mismatch.")
            source = metadata["encoder_source"]
            if mode != "scratch":
                provenance = source["provenance"]
                if (provenance.get("partition_sha256") != manifest.get("partition_sha256")
                        or provenance.get("test_used") is not False or provenance.get("pretraining_split") != "train"
                        or provenance.get("checkpoint_selection_split") != "validation"):
                    raise ValueError("Incompatible pretrained encoder provenance in resume checkpoint.")
                original = Path(config["downstream"]["encoder_checkpoint"]).expanduser()
                if str(original.resolve()) != source["path"] or (original.exists() and sha256_file(original) != source["sha256"]):
                    raise ValueError("Original pretrained encoder source/fingerprint changed.")
            elif source is not None:
                raise ValueError("Scratch resume must not have a pretrained encoder source.")
        model.load_state_dict(saved["model_state_dict"], strict=True)
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError("Nonfinite model weights in resume checkpoint.")
        optimizer.load_state_dict(state)
        model.train()
        if model.freeze_encoder:
            if (any(p.requires_grad for p in model.encoder.parameters())
                    or any(m.training for m in model.encoder.modules())
                    or _initial_state_digest(model.encoder) != metadata["initial_encoder_sha256"]):
                raise ValueError("Linear probe resume changed encoder weights/freezing/eval mode.")
        if set(saved["rng"]["loaders"]) != set(loaders):
            raise ValueError("Incomplete DataLoader RNG state.")
        if device.type in {"cuda", "mps"} and device.type not in saved["rng"]:
            raise ValueError("Missing accelerator RNG state.")
        _restore_rng(saved["rng"], device, loaders)
    except (KeyError, TypeError, RuntimeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError(f"Invalid/incomplete classifier/optimizer/RNG resume state: {exc}") from exc


def _validate_resume_binding(saved, manifest, source_hashes, output):
    best = saved["best_checkpoint"]
    if (saved["dataset_manifest_sha256"] != _manifest_digest(manifest)
            or best["dataset_manifest_sha256"] != _manifest_digest(manifest)
            or saved["partition_sha256"] != manifest.get("partition_sha256")
            or saved["source_sha256"] != source_hashes or best["label_map"] != manifest["label_map"]):
        raise ValueError("Resume dataset/partition manifest, source contents or label map changed.")
    stored = json.loads((output / "dataset_manifest.json").read_text())
    if _manifest_digest(stored) != saved["dataset_manifest_sha256"]:
        raise ValueError("Saved resume dataset manifest was modified.")
    if manifest.get("version") == 2 and load_partition_manifest(output / "partitions.json")["sha256"] != saved["partition_sha256"]:
        raise ValueError("Saved resume partitions were modified.")
    if (output / "checkpoint.pt").exists():
        current = _load_training_state(output / "checkpoint.pt")
        required = {"config", "model_state_dict", "best_epoch", "best_validation_loss"}
        if required - current.keys():
            raise ValueError("Existing best checkpoint is incomplete; cannot silently overwrite it.")
        if (current.get("format_version") != best["format_version"]
                or current.get("initial_state_sha256") != best["initial_state_sha256"]
                or current.get("dataset_manifest_sha256") != saved["dataset_manifest_sha256"]
                or _resume_protocol(current["config"]) != saved["protocol"]
                or current.get("best_epoch", math.inf) > best["best_epoch"]):
            raise ValueError("Existing best checkpoint is incompatible with resume state.")
        weights, expected = current["model_state_dict"], best["model_state_dict"]
        if (not isinstance(weights, dict) or not isinstance(expected, dict)
                or weights.keys() != expected.keys() or any(
                not isinstance(v, torch.Tensor) or v.shape != expected[k].shape
                or v.dtype != expected[k].dtype or not torch.isfinite(v).all() for k, v in weights.items())):
            raise ValueError("Existing best checkpoint has invalid model weights.")
        epoch = current["best_epoch"]
        if (type(epoch) is not int or epoch < 1
                or current["best_validation_loss"] != saved["history"][epoch - 1]["validation"]["loss"]):
            raise ValueError("Existing best checkpoint differs from saved validation history.")
        if current["best_epoch"] == best["best_epoch"] and state_digest(current["model_state_dict"]) != state_digest(best["model_state_dict"]):
            raise ValueError("Existing best checkpoint weights differ from the committed resume snapshot.")


def _archive_run(output):
    """Preserve previous/partial evaluations and scores; never delete user outputs."""
    root = output / "resume_archive"
    root.mkdir(exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix="run-", dir=root))
    for name in ("checkpoint.pt", "last_checkpoint.pt", "history.json", "resolved_config.yaml",
                 "dataset_manifest.json", "partitions.json", "preflight.json"):
        if (output / name).exists():
            shutil.copy2(output / name, archive / name)
    for name in ("scores", "metrics.json", "report.html", "plots", "tables"):
        if (output / name).exists():
            (output / name).rename(archive / name)
    print(f"Previous/partial evaluation preserved: {archive}", flush=True)


def run_classification(config: dict[str, Any], *, inspect_only: bool = False, resume=None) -> dict[str, Any]:
    """Train or resume at an epoch boundary; select exclusively on validation."""
    output = Path(config["output_dir"]).expanduser()
    if resume and inspect_only:
        raise ValueError("Resume is supported only for training, not inspection.")
    saved, resume_source = _read_resume(resume, output, config) if resume else (None, None)
    if not resume and any((output / name).exists() for name in (
            "checkpoint.pt", "last_checkpoint.pt", "history.json", "metrics.json", "scores")):
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}. Use --resume or a new --output-dir.")
    device = resolve_device(config["training"]["device"]) if not inspect_only else torch.device("cpu")
    runtime = _runtime(device)
    if saved and saved["runtime"] != runtime:
        raise ValueError("Resume runtime differs (device/software/platform); use the original environment.")
    print("Building dataset manifest and checking Parquet schemas...", flush=True)
    reader = {key: value for key, value in _reader_settings(config).items() if key != "read_batch_size"}
    partitions = None
    if config.get("partition_manifest"):
        partitions = load_partition_manifest(config["partition_manifest"])
        for key in ("validation_fraction", "test_fraction"):
            if config["data"][key] != partitions[key]:
                raise ValueError(f"{key} differs from the saved partition manifest.")
        manifest = bind_partition_manifest(partitions, config["source"], **reader)
    else:
        manifest = build_manifest(config["source"], seed=config["seed"],
                                  validation_fraction=config["data"]["validation_fraction"],
                                  test_fraction=config["data"]["test_fraction"], **reader)
    _binary_settings(config, manifest)
    source_hashes = {f["path"]: f.get("sha256") or sha256_file(f["path"])
                     for process in manifest["processes"] for f in process["files"]}
    if saved:
        _validate_resume_binding(saved, manifest, source_hashes, output)
        preflight = json.loads((output / "preflight.json").read_text())
    else:
        if inspect_only and config.get("downstream"):
            _downstream_metadata(_make_model(config, manifest["label_map"]), config, manifest)
        preflight = inspect_dataset(manifest, audit_events_per_class=config["data"]["audit_events_per_class"],
                                    **_reader_settings(config))
    print(f"Classes: {manifest['label_map']}; device: {device}", flush=True)
    for warning in preflight.get("warnings", []):
        print(f"Preflight: {warning}", flush=True)

    seed, settings = config["seed"], config["training"]
    model = optimizer = None
    if not inspect_only:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        torch.set_num_threads(settings["cpu_threads"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        # Resume loads the downstream state, never reimports the original pretrained weights.
        model = _make_model(config, manifest["label_map"], initialize_encoder=saved is None).to(device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        loaders = {name: _loader(manifest, config, name) for name in ("train", "validation")}
        if saved:
            _validate_restored_training(model, optimizer, saved, config, manifest, device, loaders)
            _archive_run(output)
            print(f"Resuming after completed epoch {saved['epoch']}; total requested epochs: {settings['epochs']}.", flush=True)

    # All resume checks succeeded before any existing artifacts are replaced.
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if partitions is not None:
        _write_json(output / "partitions.json", partitions)
    _write_json(output / "dataset_manifest.json", manifest)
    _write_json(output / "preflight.json", preflight)
    if inspect_only:
        return preflight

    start = saved["epoch"] if saved else 0
    history = copy.deepcopy(saved["history"]) if saved else []
    best_loss, best_epoch = (saved["best_loss"], saved["best_epoch"]) if saved else (math.inf, 0)
    best_payload = copy.deepcopy(saved["best_checkpoint"]) if saved else None
    diagnostics = copy.deepcopy(saved["diagnostics"]) if saved else {
        "scope": "all selected rows once: first train epoch, final validation and test"}
    first_train = copy.deepcopy(saved["first_train"]) if saved else None
    metrics = _base_metrics(config, manifest, preflight, device)
    if saved:
        metrics.update(copy.deepcopy(saved["initial_metadata"]))
    else:
        metrics.update(initialization="random_from_scratch", initial_state_sha256=_initial_state_digest(model))
        if config.get("downstream"):
            metrics["downstream"] = _downstream_metadata(model, config, manifest)
            metrics["input_contract"] = encoder_contract(config["model"], config["data"])
            if config["downstream"]["mode"] != "scratch":
                metrics["initialization"] = "pretrained_encoder_new_head"
    metrics["training_protocol"] = _training_protocol(config)

    def commit_epoch(epoch):
        # last_checkpoint is the authoritative atomic commit. It contains a recoverable
        # best snapshot if interrupted before checkpoint.pt/history.json are published.
        metrics["training_run"] = _training_run(saved, resume_source, start, epoch, config, runtime)
        best_payload.update(config=copy.deepcopy(config), training_protocol=_training_protocol(config),
                            training_run=metrics["training_run"])
        last = {"kind": "flowyforge.classification_resume", "version": 1, "epoch": epoch,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_parameters": [n for n, p in model.named_parameters() if p.requires_grad],
                "best_checkpoint": best_payload, "best_epoch": best_epoch, "best_loss": best_loss,
                "history": history, "first_train": first_train, "diagnostics": diagnostics,
                "initial_metadata": {k: metrics[k] for k in ("initialization", "initial_state_sha256", "downstream", "input_contract") if k in metrics},
                "config": config, "protocol": _resume_protocol(config), "input_contract": _classification_contract(config),
                "dataset_manifest_sha256": _manifest_digest(manifest), "partition_sha256": manifest.get("partition_sha256"),
                "source_sha256": source_hashes, "rng": _rng_state(device, loaders),
                "runtime": runtime, "training_run": metrics["training_run"]}
        last["state_sha256"] = state_digest(last)
        _save_checkpoint(output / "last_checkpoint.pt", last)
        _save_checkpoint(output / "checkpoint.pt", best_payload)
        _write_json(output / "history.json", history)

    for epoch in range(start + 1, settings["epochs"] + 1):
        started = time.monotonic()
        loaders["train"].dataset.set_epoch(epoch)
        train = _run_epoch(model, loaders["train"], device, manifest["label_map"], optimizer=optimizer,
                           gradient_clip_norm=settings["gradient_clip_norm"], log_every_batches=settings["log_every_batches"],
                           description=f"epoch {epoch} train", diagnostics=diagnostics if epoch == 1 else None)
        if first_train is None:
            first_train = train
        validation = _run_epoch(model, loaders["validation"], device, manifest["label_map"],
                                log_every_batches=settings["log_every_batches"], description=f"epoch {epoch} validation")
        if model.freeze_encoder and _initial_state_digest(model.encoder) != metrics["downstream"]["initial_encoder_sha256"]:
            raise ValueError("Linear probe changed its frozen encoder during training.")
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                        "train_num_events": train["num_events"], "train_duration_seconds": train["duration_seconds"],
                        "train_events_per_second": train["events_per_second"], "validation": validation,
                        "duration_seconds": time.monotonic() - started})
        print(f"Epoch {epoch}/{settings['epochs']}: train loss={train['loss']:.5f}, "
              f"validation loss={validation['loss']:.5f}, accuracy={validation['accuracy']:.4f}, "
              f"macro F1={validation['macro_f1']:.4f}", flush=True)
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            if config.get("downstream"):
                metrics["downstream"]["selected_encoder_sha256"] = _initial_state_digest(model.encoder)
                metrics["downstream"]["selected_head_sha256"] = _initial_state_digest(model.head)
            best_payload = {"format_version": 2 if config.get("downstream") else 1,
                        **{key: copy.deepcopy(metrics[key]) for key in ("downstream", "input_contract") if key in metrics},
                        "best_validation_loss": best_loss,
                        "model_state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                        "config": copy.deepcopy(config), "label_map": manifest["label_map"], "best_epoch": best_epoch,
                        **{key: metrics[key] for key in ("initialization", "initial_state_sha256", "training_protocol")},
                        "dataset_manifest_sha256": metrics["dataset_manifest_sha256"],
                        "collection_feature_dims": collection_feature_dims(default_collection_specs(config["data"]["collection_limits"]))}
        commit_epoch(epoch)

    if start == settings["epochs"]:
        commit_epoch(start)  # Retry interrupted final evaluation with no optimizer updates.
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    metrics.update(history=history, best_epoch=best_epoch, best_validation_loss=best_loss,
                   performance=_performance(history), diagnostics=diagnostics)
    metrics["checkpoint_sha256"] = sha256_file(output / "checkpoint.pt")
    # Test is constructed and iterated only after all optimization/selection is complete.
    loaders["test"] = _loader(manifest, config, "test")
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
    if any((output / name).exists() for name in ("checkpoint.pt", "last_checkpoint.pt", "history.json", "scores")):
        raise FileExistsError(f"Evaluation output contains an existing training run: {output}. Choose a separate directory.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in {1, 2}:
        raise ValueError("Expected a streaming classification checkpoint (format_version=1 or 2).")
    manifest = json.loads((checkpoint_path.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
    if _manifest_digest(manifest) != checkpoint["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the manifest saved with the checkpoint.")
    config = checkpoint["config"]
    if checkpoint["format_version"] == 2:
        if (not config.get("downstream") or checkpoint.get("input_contract") != encoder_contract(config["model"], config["data"])
            or checkpoint.get("downstream", {}).get("mode") != config["downstream"]["mode"]):
            raise ValueError("Incompatible downstream checkpoint contract/configuration.")
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
    model = _make_model(config, manifest["label_map"], initialize_encoder=False).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    preflight = json.loads((checkpoint_path.parent / "preflight.json").read_text(encoding="utf-8"))
    metrics = _base_metrics(config, manifest, preflight, resolved_device)
    metrics.update({key: checkpoint.get(key) for key in ("initialization", "initial_state_sha256", "training_protocol")})
    metrics.update({key: checkpoint[key] for key in ("downstream", "input_contract", "best_validation_loss", "training_run") if key in checkpoint})
    metrics["checkpoint_sha256"] = sha256_file(checkpoint_path)
    history_path = checkpoint_path.parent / "history.json"
    metrics.update(best_epoch=checkpoint["best_epoch"], history=json.loads(history_path.read_text()) if history_path.exists() else [],
                   checkpoint_path=str(checkpoint_path.resolve()), diagnostics={"scope": "validation and test rows only"})
    metrics["performance"] = _performance(metrics["history"])
    for split in ("validation", "test"):
        metrics[split] = _evaluate_split(model, _loader(manifest, config, split), resolved_device,
                                        manifest, config, split, output, metrics)
    _record_split_counts(metrics)
    _finish_diagnostics(metrics["diagnostics"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if manifest.get("version") == 2:
        _write_json(output / "dataset_manifest.json", manifest)
        _write_json(output / "partitions.json", partitions)
    _write_json(output / "metrics.json", metrics)
    write_classification_report(metrics, output)
    return metrics


__all__ = ["load_run_config", "resolve_device", "batch_to_device", "run_classification", "evaluate_run"]
