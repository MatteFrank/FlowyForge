"""Masked reconstruction on shared train/validation partitions; never iterate test."""

from __future__ import annotations

import copy
import math
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml

from flowyforge.training.checkpointing import _rng_state, _restore_rng, _save_checkpoint
from flowyforge.core.config import load_yaml
from flowyforge.data_plugins.collide_v2.streaming import bind_partition_manifest, load_partition_manifest
from flowyforge.model_plugins.foundation_encoder import (
    FoundationEncoder, MaskedReconstructionModel, encoder_contract, export_encoder,
)
from flowyforge.model_plugins.perceiver import PerceiverConfig
from flowyforge.task_plugins.masked_modeling import MaskedModelingTask
from flowyforge.training.classification_run import (
    _known, _loader, _manifest_digest, _positive_integer, _reader_settings, _resolve_run_config,
    _write_json, batch_to_device, resolve_device,
)


def load_pretraining_config(path, overrides=None):
    path = Path(path).expanduser().resolve()
    raw = load_yaml(path)
    from flowyforge.analysis.pretraining_report import report_options
    reporting = report_options(raw.pop("reporting", {}))
    masking = {"probability": 0.15, "seed": 12345, "huber_delta": 1.0, "decoder_hidden_dim": 64,
               "targets": None, "collection_weights": {}, "feature_weights": {}, **raw.pop("pretraining", {})}
    _known(masking, {"probability", "seed", "huber_delta", "decoder_hidden_dim", "targets",
                     "collection_weights", "feature_weights"}, "pretraining")
    _positive_integer(masking["decoder_hidden_dim"], "decoder_hidden_dim")
    config = _resolve_run_config(raw, path, overrides)
    if not config["partition_manifest"]:
        raise ValueError("Pretraining requires the shared partition_manifest; create it with run_classification.py --stage partitions.")
    config.pop("evaluation")  # Supervised evaluation settings have no role in masking.
    config["pretraining"] = masking
    config["reporting"] = reporting
    task = _task(config)
    config["pretraining"]["targets"] = task.targets
    return config


def _task(config):
    contract = encoder_contract(config["model"], config["data"])
    features = {c["name"]: c["feature_names"] for c in contract["collections"]}
    return MaskedModelingTask(features, **{k: v for k, v in config["pretraining"].items() if k != "decoder_hidden_dim"})


def _model(config, task):
    contract = encoder_contract(config["model"], config["data"])
    dims = {c["name"]: len(c["feature_names"]) for c in contract["collections"]}
    encoder = FoundationEncoder(dims, PerceiverConfig(**contract["architecture"]), mask_indicators=True)
    return MaskedReconstructionModel(encoder, task.targets, config["pretraining"]["decoder_hidden_dim"])


def _accumulate(totals, statistics):
    for key, (total, count) in statistics.items():
        previous, seen = totals.get(key, (0.0, 0))
        totals[key] = (previous + float(total.detach()) if isinstance(total, torch.Tensor) else previous + total, seen + count)


def _summary(task, totals):
    return {"loss": task.combine(totals), "features": {
        key: {"huber": total / count, "masked_count": count} for key, (total, count) in totals.items()}}


def fit_training_constant(loader, task):
    """Streaming feature means in transformed units, from eligible TRAIN values only."""
    totals = {}
    for batch in loader:
        targets, valid = task.eligible(batch)
        for name, values in targets.items():
            for index, feature in enumerate(task.targets[name]):
                selected = values[..., index][valid[name][..., index]].double()
                if selected.numel():
                    _accumulate(totals, {task.key(name, feature): (selected.sum(), selected.numel())})
    if not totals:
        raise ValueError("Training partition has no eligible reconstruction targets.")
    return {"estimator": "arithmetic mean of eligible transformed training values", "split": "train",
            "features": {key: {"value": total / count, "count": count} for key, (total, count) in totals.items()}}


def _constant_predictions(prepared, task, baseline):
    outputs = {}
    for name, target in prepared.targets.items():
        values = []
        for index, feature in enumerate(task.targets[name]):
            key = task.key(name, feature)
            estimate = baseline["features"].get(key)
            if estimate is None and prepared.masks[name][..., index].any():
                raise ValueError(f"No training-only constant available for validation target {key}.")
            values.append(estimate["value"] if estimate else 0.0)
        outputs[name] = target.new_tensor(values).expand_as(target)
    return outputs


def _epoch(model, loader, task, device, *, epoch, baseline, optimizer=None, gradient_clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    totals, constant_totals = {}, {}
    counts = {"num_events": 0, "empty_events": 0, "batches": 0, "skipped_batches": 0, "optimizer_steps": 0}
    with torch.set_grad_enabled(training):
        for raw in loader:
            batch = batch_to_device(raw, device)
            counts["num_events"] += batch.metadata["batch_size"]
            counts["batches"] += 1
            present = torch.zeros(batch.metadata["batch_size"], dtype=torch.bool, device=device)
            for group in batch.objects.values():
                present |= group.mask.any(dim=1)
            counts["empty_events"] += int((~present).sum())
            prepared = task.prepare_batch(batch, epoch=epoch, validation=not training)
            if not any(mask.any() for mask in prepared.masks.values()):
                counts["skipped_batches"] += 1
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(prepared.inputs)
            statistics = task.loss_statistics(predictions, prepared)
            loss = task.combine(statistics)
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError("Masked reconstruction loss is not finite.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                counts["optimizer_steps"] += 1
            else:
                _accumulate(constant_totals, task.loss_statistics(_constant_predictions(prepared, task, baseline), prepared))
            _accumulate(totals, statistics)
    result = {**counts, **_summary(task, totals)}
    if not training:
        result["training_constant"] = _summary(task, constant_totals)
        result["improvement_over_constant"] = (result["training_constant"]["loss"] - result["loss"]
                                                if result["loss"] is not None else None)
    return result


def _resume_protocol(config):
    protocol = copy.deepcopy(config)
    protocol.pop("reporting", None)  # Diagnostics never affect optimization or resume compatibility.
    for key in ("config_path", "model_config_path", "output_dir"):
        protocol.pop(key)
    # Epoch-boundary resume; only extend the horizon or change runtime device.
    protocol["training"].pop("epochs")
    protocol["training"].pop("device")
    return protocol


def run_pretraining(config, *, inspect_only=False, resume=None):
    output = Path(config["output_dir"]).expanduser()
    saved = None
    if resume:
        if inspect_only or Path(resume).resolve() != (output / "checkpoint.pt").resolve():
            raise ValueError("Resume requires checkpoint.pt in the same output directory and stage train.")
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("kind") != "flowyforge.masked_pretraining" or saved.get("version") != 1:
            raise ValueError("Unsupported pretraining checkpoint kind/version.")
        if saved["protocol"] != _resume_protocol(config):
            raise ValueError("Incompatible resume configuration: only epochs/device may change.")
        if config["training"]["epochs"] <= saved["epoch"]:
            raise ValueError("Resume epochs is the TOTAL horizon and must exceed the saved epoch.")
    elif any((output / name).exists() for name in ("checkpoint.pt", "encoder.pt", "history.json", "metrics.json")):
        raise FileExistsError(f"Refusing to overwrite pretraining run {output}; use --resume or another --output-dir.")

    task = _task(config)
    contract = encoder_contract(config["model"], config["data"])
    partitions = load_partition_manifest(config["partition_manifest"])
    for key in ("validation_fraction", "test_fraction"):
        if config["data"][key] != partitions[key]:
            raise ValueError(f"{key} differs from the shared partition manifest.")
    reader = {k: v for k, v in _reader_settings(config).items() if k != "read_batch_size"}
    manifest = bind_partition_manifest(partitions, config["source"], **reader)
    digest = _manifest_digest(manifest)
    if saved and (saved["dataset_sha256"] != digest or saved["contract"] != contract):
        raise ValueError("Incompatible pretraining checkpoint: dataset manifest or encoder contract changed.")
    # No test loader, test loss or test baseline, including in inspection.
    loaders = {"train": _loader(manifest, config, "train", purpose="pretraining"),
               "validation": _loader(manifest, config, "validation", purpose="checkpoint_selection")}
    settings = config["training"]
    device = resolve_device(settings["device"]) if not inspect_only else torch.device("cpu")
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"] % 2**32)
    model = _model(config, task).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    baseline = saved["constant_baseline"] if saved else fit_training_constant(loaders["train"], task)
    preflight = {"view": config["data"]["view"], "partition_sha256": partitions["sha256"],
                 "split_counts": manifest["split_counts"], "test_used": False,
                 "contract": contract, "constant_baseline": baseline}
    if inspect_only:
        # Check transformed validation inputs/masks without running a training epoch.
        totals = {}
        for batch in loaders["validation"]:
            prepared = task.prepare_batch(batch, validation=True)
            _accumulate(totals, task.loss_statistics(_constant_predictions(prepared, task, baseline), prepared))
        preflight["validation_constant"] = _summary(task, totals)
        if preflight["validation_constant"]["loss"] is None:
            raise ValueError("Validation has no masked targets: enlarge the sample or increase masking probability.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for name, payload in (("partitions.json", partitions), ("dataset_manifest.json", manifest), ("preflight.json", preflight)):
        _write_json(output / name, payload)
    if inspect_only:
        return preflight

    history, best_loss, best_epoch, best_state, start = [], math.inf, 0, None, 0
    best_model_state = None
    if saved:
        try:
            model.load_state_dict(saved["model_state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
        except (KeyError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Incompatible pretraining model/optimizer state: {exc}") from exc
        history, best_loss, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        best_state, start = saved["best_encoder_state_dict"], saved["epoch"]
        from flowyforge.analysis.pretraining_report import selected_model_state
        best_model_state, _ = selected_model_state(saved)
        _restore_rng(saved["rng"], device, loaders)
    for epoch in range(start + 1, settings["epochs"] + 1):
        loaders["train"].dataset.set_epoch(epoch)
        training = _epoch(model, loaders["train"], task, device, epoch=epoch, baseline=baseline,
                          optimizer=optimizer, gradient_clip_norm=settings["gradient_clip_norm"])
        if training["loss"] is None:
            raise ValueError("Training epoch has no masked targets; enlarge sample or increase masking probability.")
        validation = _epoch(model, loaders["validation"], task, device, epoch=epoch, baseline=baseline)
        if validation["loss"] is None:
            raise ValueError("Validation has no masked targets; checkpoint selection is undefined.")
        history.append({"epoch": epoch, "train": training, "validation": validation})
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        checkpoint = {"kind": "flowyforge.masked_pretraining", "version": 1, "epoch": epoch,
                      "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                      "best_encoder_state_dict": best_state, "best_epoch": best_epoch, "best_loss": best_loss,
                      "best_model_state_dict": best_model_state,
                      "best_model_epoch": best_epoch if best_model_state is not None else None,
                      "history": history, "constant_baseline": baseline, "contract": contract,
                      "config": config, "protocol": _resume_protocol(config), "dataset_sha256": digest,
                      "partition_sha256": partitions["sha256"], "rng": _rng_state(device, loaders)}
        _save_checkpoint(output / "checkpoint.pt", checkpoint)
        # Export selected encoder independently of the current/last model state.
        selected = copy.deepcopy(model.encoder).cpu()
        selected.load_state_dict(best_state, strict=True)
        export_encoder(output / "encoder.pt", selected, contract,
                       {"partition_manifest": "partitions.json", "partition_sha256": partitions["sha256"],
                        "dataset_manifest": "dataset_manifest.json", "dataset_sha256": digest,
                        "input_partition_manifest": config["partition_manifest"], "best_epoch": best_epoch,
                        "pretraining_split": "train", "checkpoint_selection_split": "validation", "test_used": False})
        metrics = {"kind": "masked_pretraining", "version": 1, "view": config["data"]["view"],
                   "best_epoch": best_epoch, "validation": history[best_epoch - 1]["validation"],
                   "history": history, "constant_baseline": baseline, "test_used": False,
                   "partition_sha256": partitions["sha256"], "encoder": "encoder.pt"}
        _write_json(output / "history.json", history)
        _write_json(output / "metrics.json", metrics)
        print(f"Epoch {epoch}: train Huber={training['loss']:.6g}, validation={validation['loss']:.6g}, "
              f"training constant={validation['training_constant']['loss']:.6g}; best={best_epoch}", flush=True)
    # Reporting is post-training, uses a separate model, and cannot change selection/resume.
    from flowyforge.analysis.pretraining_report import generate_pretraining_report
    try:
        report = generate_pretraining_report(output, options=config.get("reporting"), overwrite=True)
        print(f"Pretraining diagnostics: {report['html']}", flush=True)
        if report["inference"].get("reason"):
            print(f"Diagnostic limitation: {report['inference']['reason']}", flush=True)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError, FloatingPointError) as exc:
        warnings.warn(f"Training saved successfully; pretraining report failed: {exc}. "
                      "Retry scripts/report_pretraining.py on this run.", RuntimeWarning)
    return metrics
