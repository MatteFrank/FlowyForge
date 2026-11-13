"""Bounded, validation-only diagnostics for masked pretraining (no optimization)."""

from __future__ import annotations

import copy
import csv
import hashlib
import heapq
import html
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

KIND = "flowyforge.pretraining_report"
DEFAULTS = {"inference": True, "max_events": 512, "batch_size": 16, "seed": 2718,
            "bins": 24, "min_bin_count": 20}


def report_options(options=None):
    options = dict(options or {})
    if set(options) - DEFAULTS.keys():
        raise ValueError(f"Unknown reporting options: {sorted(set(options) - DEFAULTS.keys())}")
    result = {**DEFAULTS, **options}
    if type(result["inference"]) is not bool:
        raise ValueError("reporting.inference must be boolean.")
    for key in ("max_events", "batch_size", "bins", "min_bin_count"):
        if type(result[key]) is not int or result[key] < (2 if key in {"bins", "min_bin_count"} else 1):
            raise ValueError(f"reporting.{key} must be a positive integer (bins/min_bin_count >= 2).")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("reporting.seed must be a nonnegative integer.")
    return result


def selected_model_state(checkpoint):
    """Never combine a selected encoder with a decoder from the last epoch."""
    if checkpoint.get("kind") != "flowyforge.masked_pretraining" or checkpoint.get("version") != 1:
        raise ValueError("Unsupported masked pretraining checkpoint kind/version.")
    state = checkpoint.get("best_model_state_dict")
    if state is not None:
        if checkpoint.get("best_model_epoch") != checkpoint["best_epoch"]:
            raise ValueError("Best full-model epoch differs from selected encoder epoch.")
        source = "best_model_state_dict"
    elif checkpoint["best_epoch"] == checkpoint["epoch"]:
        state, source = checkpoint["model_state_dict"], "legacy_last_equals_best"
    else:
        return None, "Legacy checkpoint has no decoder for the best epoch; reconstruction unavailable."
    encoder = {key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")}
    expected = checkpoint["best_encoder_state_dict"]
    if encoder.keys() != expected.keys() or any(not torch.equal(encoder[k], expected[k]) for k in expected):
        raise ValueError("Selected full-model encoder differs from best_encoder_state_dict.")
    return state, source


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json(path, value):
    path.write_text(json.dumps(_clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows, fields=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(_clean(rows))


def _figure(rows=1, cols=1, **kwargs):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(6 * cols, 3.8 * rows), constrained_layout=True)
    FigureCanvasAgg(fig)
    return fig, fig.subplots(rows, cols, squeeze=False, **kwargs)


def _save(fig, directory, name, figures):
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", dpi=125)
    figures.append(name)
    fig.clear()


def _history_report(history, directory, figures):
    rows, counts = [], []
    for epoch in history:
        for split in ("train", "validation"):
            record = epoch[split]
            constant = record.get("training_constant", {})
            counts.append({"epoch": epoch["epoch"], "split": split, "loss": record.get("loss"),
                           "constant_huber": constant.get("loss"),
                           "masked_targets": sum(f.get("masked_count", 0) for f in record.get("features", {}).values()),
                           **{k: record.get(k) for k in ("num_events", "empty_events", "batches", "skipped_batches", "optimizer_steps")}})
            for feature, stats in record.get("features", {}).items():
                base = constant.get("features", {}).get(feature, {}).get("huber")
                loss = stats.get("huber")
                relative = (base - loss) / base if base is not None and base > 0 and loss is not None else None
                rows.append({"epoch": epoch["epoch"], "split": split, "collection_feature": feature,
                             "huber": loss, "constant_huber": base, "relative_improvement": relative,
                             "masked_count": stats.get("masked_count", 0)})
    _csv(directory / "history.csv", counts)
    _csv(directory / "feature_history.csv", rows,
         ["epoch", "split", "collection_feature", "huber", "constant_huber", "relative_improvement", "masked_count"])
    fig, axes = _figure()
    for split in ("train", "validation"):
        records = [r for r in counts if r["split"] == split]
        axes[0, 0].plot([r["epoch"] for r in records], [r["loss"] for r in records], "o-", label=split)
    axes[0, 0].plot([e["epoch"] for e in history],
                    [e["validation"].get("training_constant", {}).get("loss") for e in history],
                    "--", label="train constant evaluated on validation")
    axes[0, 0].set(xlabel="Epoch", ylabel="Balanced Huber (transformed units)")
    axes[0, 0].legend(fontsize=8)
    _save(fig, directory, "loss", figures)
    keys = sorted({r["collection_feature"] for r in rows})
    if keys:
        fig, axes = _figure()
        values = np.full((len(keys), len(history)), np.nan)
        for j, epoch in enumerate(history):
            for r in rows:
                if r["epoch"] == epoch["epoch"] and r["split"] == "validation" and r["relative_improvement"] is not None:
                    values[keys.index(r["collection_feature"]), j] = r["relative_improvement"]
        finite = np.abs(values[np.isfinite(values)])
        bound = max(float(finite.max()) if len(finite) else 1., 1.e-8)
        im = axes[0, 0].imshow(np.ma.masked_invalid(values), aspect="auto", cmap="RdBu", vmin=-bound, vmax=bound)
        axes[0, 0].set(yticks=range(len(keys)), yticklabels=keys,
                       xticks=range(len(history)), xticklabels=[e["epoch"] for e in history], xlabel="Epoch")
        axes[0, 0].tick_params(axis="y", labelsize=8)
        fig.colorbar(im, ax=axes[0, 0], label="(constant − model) / constant; validation")
        _save(fig, directory, "improvement", figures)
    fig, axes = _figure(1, 3)
    for split in ("train", "validation"):
        records = [r for r in counts if r["split"] == split]
        for ax, field in zip(axes[0], ("masked_targets", "skipped_batches", "empty_events")):
            ax.plot([r["epoch"] for r in records], [r[field] for r in records], "o-", label=split)
            ax.set(xlabel="Epoch", ylabel=field)
            ax.legend()
    _save(fig, directory, "counts", figures)


def _sample_ids(partitions, limit, seed):
    # Selection depends only on original identity and seed, never on labels/view/order.
    ids = (e["id"] for p in partitions["processes"] for e in p["events"] if e["split"] == "validation")
    return heapq.nsmallest(limit, ids, key=lambda key: (hashlib.sha256(f"report-v1|{seed}|{key}".encode()).digest(), key))


def _sample_manifest(manifest, selected):
    result = copy.deepcopy(manifest)
    for process in result["processes"]:
        indices = {info["offset"] + i for info in process["files"]
                   for i, key in enumerate(info["event_ids"]) if key in selected}
        process["split_indices"]["validation"] = [i for i in process["split_indices"]["validation"] if i in indices]
        process["split_counts"]["validation"] = len(process["split_indices"]["validation"])
    result["split_counts"]["validation"] = sum(p["split_counts"]["validation"] for p in result["processes"])
    return result


def _physical(values):
    # Predictions remain unconstrained: negative values are not silently clipped.
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.expm1(np.asarray(values, dtype=np.float64))
    return np.where(np.isfinite(result), result, np.nan)


class _Moments:
    """Mergeable centered moments; memory O(D²), independent of event count."""
    def __init__(self, dim):
        self.n, self.mean, self.m2 = 0, np.zeros(dim), np.zeros((dim, dim))

    def add(self, values):
        if not len(values):
            return
        n, mean = len(values), values.mean(axis=0)
        centered = values - mean
        delta = mean - self.mean
        self.m2 += centered.T @ centered + np.outer(delta, delta) * (self.n * n / (self.n + n))
        self.mean += delta * (n / (self.n + n))
        self.n += n


PREDICTION_SCHEMA = pa.schema([
    ("event_id", pa.string()), ("split", pa.string()), ("view", pa.string()), ("epoch", pa.int64()),
    ("collection", pa.string()), ("feature", pa.string()), ("object_index", pa.int64()), ("object_slot", pa.int64()),
    *[(f"{name}_{unit}", pa.float64()) for unit in ("transformed", "GeV") for name in ("true", "predicted", "baseline")],
])


def _infer(checkpoint, model_state, manifest, partitions, options, directory):
    from flowyforge.training.classification_run import _loader
    from flowyforge.training.pretraining_run import _model, _task, _accumulate, _summary, _constant_predictions

    config = copy.deepcopy(checkpoint["config"])
    config["training"].update(batch_size=options["batch_size"], num_workers=0)
    selected = set(_sample_ids(partitions, options["max_events"], options["seed"]))
    task = _task(config)
    model = _model(config, task).cpu().eval()
    model.load_state_dict(model_state, strict=True)
    baseline = checkpoint["constant_baseline"]
    if baseline.get("split") != "train":
        raise ValueError("Report baseline must have been estimated on training only.")
    loader = _loader(_sample_manifest(manifest, selected), config, "validation", purpose="checkpoint_selection")
    moments = _Moments(model.encoder.output_dim)
    totals, constant_totals, seen, ranges = {}, {}, set(), {}
    counts = {"events": 0, "empty_events": 0, "batches": 0, "batches_without_masked_targets": 0,
              "masked_targets": 0, "nonfinite_embeddings": 0, "nonfinite_predictions": 0,
              "physical_overflow": 0, "negative_physical_predictions": 0}
    embedding_schema = pa.schema([("event_id", pa.string()), ("split", pa.string()), ("empty", pa.bool_()),
                                  ("embedding", pa.list_(pa.float64(), model.encoder.output_dim))])
    metadata = {b"split": b"validation", b"best_epoch": str(checkpoint["best_epoch"]).encode(),
                b"partition_sha256": checkpoint["partition_sha256"].encode()}
    with (pq.ParquetWriter(directory / "predictions.parquet", PREDICTION_SCHEMA.with_metadata(metadata)) as writer,
          pq.ParquetWriter(directory / "embeddings.parquet", embedding_schema.with_metadata(metadata)) as ew,
          torch.inference_mode()):
        for batch in loader:
            events = batch.metadata["events"]
            ids = [e["event_id"] for e in events]
            if len(set(ids)) != len(ids) or set(ids) & seen or not set(ids) <= selected:
                raise ValueError("Duplicated or unexpected validation event identity in diagnostic stream.")
            seen.update(ids)
            counts["events"] += len(ids)
            counts["batches"] += 1
            present = torch.zeros(len(ids), dtype=torch.bool)
            for group in batch.objects.values():
                present |= group.mask.any(dim=1)
            counts["empty_events"] += int((~present).sum())
            # Clean-input embeddings describe the reusable encoder, separately from masked reconstruction.
            embeddings = model.encoder.encode(batch).double().numpy()
            finite_embeddings = np.isfinite(embeddings).all(axis=1)
            counts["nonfinite_embeddings"] += int((~finite_embeddings).sum())
            moments.add(embeddings[finite_embeddings & present.numpy()])
            ew.write_table(pa.Table.from_pylist([
                {"event_id": key, "split": "validation", "empty": not bool(present[i]),
                 "embedding": _clean(embeddings[i])} for i, key in enumerate(ids)], schema=embedding_schema))
            prepared = task.prepare_batch(batch, validation=True)
            if not any(mask.any() for mask in prepared.masks.values()):
                counts["batches_without_masked_targets"] += 1
                continue
            predictions = model(prepared.inputs)
            constants = _constant_predictions(prepared, task, baseline)
            _accumulate(totals, task.loss_statistics(predictions, prepared))
            _accumulate(constant_totals, task.loss_statistics(constants, prepared))
            rows = []  # At most one configured batch of scalar masked-target records.
            for collection, names in task.targets.items():
                if collection not in prepared.targets:
                    continue
                for index, feature in enumerate(names):
                    mask = prepared.masks[collection][..., index]
                    positions = mask.nonzero().tolist()
                    values = np.stack([tensor[collection][..., index][mask].double().numpy()
                                       for tensor in (prepared.targets, predictions, constants)], axis=1)
                    physical = _physical(values)
                    key = task.key(collection, feature)
                    if len(values):
                        ranges.setdefault(key, {"transformed": [math.inf, -math.inf], "GeV": [math.inf, -math.inf], "true_max": 0.})
                        for unit, array in (("transformed", values), ("GeV", physical)):
                            finite = array[np.isfinite(array)]
                            if len(finite):
                                ranges[key][unit][0] = min(ranges[key][unit][0], float(finite.min()))
                                ranges[key][unit][1] = max(ranges[key][unit][1], float(finite.max()))
                        ranges[key]["true_max"] = max(ranges[key]["true_max"], float(physical[:, 0].max()))
                    counts["nonfinite_predictions"] += int((~np.isfinite(values[:, 1])).sum())
                    counts["physical_overflow"] += int((np.isfinite(values[:, 1]) & ~np.isfinite(physical[:, 1])).sum())
                    counts["negative_physical_predictions"] += int((physical[:, 1] < 0).sum())
                    for j, (event, slot) in enumerate(positions):
                        original = batch.objects[collection].metadata["events"][event]["object_indices"][slot]
                        rows.append({"event_id": ids[event], "split": "validation", "view": config["data"]["view"],
                                     "epoch": checkpoint["best_epoch"], "collection": collection, "feature": feature,
                                     "object_index": original, "object_slot": slot,
                                     **{f"{name}_{unit}": _clean(array[j, k])
                                        for unit, array in (("transformed", values), ("GeV", physical))
                                        for k, name in enumerate(("true", "predicted", "baseline"))}})
            counts["masked_targets"] += len(rows)
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=PREDICTION_SCHEMA))
    if seen != selected:
        raise ValueError("Diagnostic stream did not cover exactly the selected validation IDs.")
    _csv(directory / "sample_events.csv", [{"event_id": key, "split": "validation"} for key in sorted(seen)], ["event_id", "split"])
    return {"counts": counts, "sample_loss": _summary(task, totals), "sample_constant": _summary(task, constant_totals),
            "sample_event_ids_sha256": hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest()}, ranges, moments


def _edges(low, high, bins):
    if low == high:
        width = max(abs(low) * .05, .5)
        low, high = low - width, high + width
    return np.linspace(low, high, bins + 1)


def _prediction_plots(directory, ranges, options, figures, target_keys):
    """Two passes over scalar Parquet; histograms/moments have fixed memory."""
    bins, minimum = options["bins"], options["min_bin_count"]
    summaries, residual_rows, histogram_rows = [], [], []
    for key in target_keys:
        if key not in ranges:
            summaries.append({"feature": key, "status": "no_masked_targets", "count": 0})
            continue
        collection, feature = key.split("/")
        spec = ranges[key]
        edges = {unit: _edges(*spec[unit], bins) for unit in ("transformed", "GeV")}
        true_edges = np.expm1(np.linspace(0., np.log1p(max(spec["true_max"], 1.e-6)), bins + 1))
        true_edges[-1] = np.nextafter(true_edges[-1], math.inf)
        hist = {unit: np.zeros((3, bins), dtype=np.int64) for unit in edges}
        pairs = {unit: np.zeros((bins, bins), dtype=np.int64) for unit in edges}
        # Per true bin, predictor and absolute/relative residual: stable centered moments.
        moments = {(b, name, residual): _Moments(1) for b in range(bins)
                   for name in ("predicted", "baseline") for residual in ("absolute", "relative")}
        total, invalid, zeros = 0, 0, 0
        with pq.ParquetFile(directory / "predictions.parquet") as parquet:
            for batch in parquet.iter_batches(batch_size=4096):
                rows = [r for r in batch.to_pylist() if r["collection"] == collection and r["feature"] == feature]
                if not rows:
                    continue
                total += len(rows)
                for unit in edges:
                    arrays = [np.array([r[f"{name}_{unit}"] if r[f"{name}_{unit}"] is not None else np.nan for r in rows])
                              for name in ("true", "predicted", "baseline")]
                    for i, values in enumerate(arrays):
                        hist[unit][i] += np.histogram(values[np.isfinite(values)], edges[unit])[0]
                    valid = np.isfinite(arrays[0]) & np.isfinite(arrays[1])
                    pairs[unit] += np.histogram2d(arrays[0][valid], arrays[1][valid], bins=(edges[unit], edges[unit]))[0].astype(np.int64)
                true, predicted, constant = arrays  # Last unit is explicitly GeV.
                invalid += int((~np.isfinite(predicted)).sum())
                zeros += int((true == 0).sum())
                assigned = np.clip(np.searchsorted(true_edges, true, side="right") - 1, 0, bins - 1)
                for b in range(bins):
                    for name, values in (("predicted", predicted), ("baseline", constant)):
                        valid = (assigned == b) & np.isfinite(true) & np.isfinite(values)
                        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                            absolute = values[valid] - true[valid]
                            relative = absolute[true[valid] > 0] / true[valid][true[valid] > 0]
                        for kind, residual in (("absolute", absolute), ("relative", relative)):
                            moments[b, name, kind].add(residual[np.isfinite(residual), None])
        summaries.append({"feature": key, "status": "ok" if total >= minimum else "insufficient_statistics",
                          "count": total, "invalid_physical_predictions": invalid, "true_zero_count": zeros})
        name = key.replace("/", "_")
        fig, axes = _figure(2, 2)
        for i, unit in enumerate(edges):
            im = axes[i, 0].pcolormesh(edges[unit], edges[unit], np.log1p(pairs[unit].T), shading="auto")
            axes[i, 0].plot([edges[unit][0], edges[unit][-1]], [edges[unit][0], edges[unit][-1]], "k--", linewidth=1)
            axes[i, 0].set(xlabel=f"True ({unit})", ylabel=f"Predicted ({unit})")
            fig.colorbar(im, ax=axes[i, 0], label="log(1 + count)")
            for j, label in enumerate(("true", "predicted", "training constant")):
                axes[i, 1].stairs(hist[unit][j], edges[unit], label=label)
            axes[i, 1].set(xlabel=unit, ylabel="Masked-target count")
            axes[i, 1].legend(fontsize=8)
            for b in range(bins):
                histogram_rows.append({"feature": key, "unit": unit, "low": edges[unit][b], "high": edges[unit][b + 1],
                                       **{label: hist[unit][j, b] for j, label in enumerate(("true", "predicted", "baseline"))}})
        fig.suptitle(f"{key} — validation, masked targets only (N={total})")
        _save(fig, directory, name + "_reconstruction", figures)
        fig, axes = _figure(2, 2)
        for i, kind in enumerate(("absolute", "relative")):
            for label in ("predicted", "baseline"):
                bias, resolution = [], []
                for b in range(bins):
                    moment = moments[b, label, kind]
                    good = moment.n >= minimum and np.isfinite(moment.m2).all() and np.isfinite(moment.mean).all()
                    mean = float(moment.mean[0]) if good else None
                    std = math.sqrt(max(float(moment.m2[0, 0]) / (moment.n - 1), 0.)) if good else None
                    residual_rows.append({"feature": key, "predictor": label, "residual": kind,
                        "true_low_GeV": true_edges[b], "true_high_GeV": true_edges[b + 1], "count": moment.n,
                        "bias": mean, "resolution_std": std, "status": "ok" if good else "insufficient_or_nonfinite"})
                    bias.append(mean if mean is not None else np.nan)
                    resolution.append(std if std is not None else np.nan)
                centers = (true_edges[:-1] + true_edges[1:]) / 2
                for ax, values, title in ((axes[i, 0], bias, "Bias: mean residual"), (axes[i, 1], resolution, "Resolution: sample std")):
                    ax.plot(centers, values, "o-", label=label)
                    ax.set(xlabel=f"True {'mass' if feature == 'log1p_Mass' else 'PT'} [GeV] (reader clipped)",
                           ylabel=f"{title} ({'GeV' if kind == 'absolute' else 'relative'})", xscale="symlog")
                    ax.legend(fontsize=8)
        fig.suptitle(f"{key} — bins require N ≥ {minimum}; relative residual excludes true=0")
        _save(fig, directory, name + "_bias_resolution", figures)
    _csv(directory / "histograms.csv", histogram_rows, ["feature", "unit", "low", "high", "true", "predicted", "baseline"])
    _csv(directory / "bias_resolution.csv", residual_rows,
         ["feature", "predictor", "residual", "true_low_GeV", "true_high_GeV", "count", "bias", "resolution_std", "status"])
    return summaries


def _embedding_report(moments, directory, figures):
    if moments.n < 2:
        return {"status": "insufficient_nonempty_finite_events", "count": moments.n}
    covariance = moments.m2 / (moments.n - 1)
    if not np.isfinite(covariance).all():
        return {"status": "nonfinite_covariance", "count": moments.n}
    spectrum = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0.)
    total = float(spectrum.sum())
    proportions = spectrum / total if total > 0 else np.zeros_like(spectrum)
    effective_rank = float(np.exp(-np.sum(proportions[proportions > 0] * np.log(proportions[proportions > 0])))) if total > 0 else 0.
    variances = np.diag(covariance)
    _csv(directory / "embedding_spectrum.csv", [{"rank": i + 1, "eigenvalue": v, "variance_fraction": proportions[i]}
                                               for i, v in enumerate(spectrum)])
    _csv(directory / "embedding_variance.csv", [{"dimension": i, "mean": moments.mean[i], "variance": v}
                                               for i, v in enumerate(variances)])
    fig, axes = _figure(1, 2)
    axes[0, 0].plot(np.arange(1, len(spectrum) + 1), spectrum, "o-")
    axes[0, 0].set(xlabel="Rank", ylabel="Centered covariance eigenvalue")
    axes[0, 1].bar(np.arange(len(variances)), variances)
    axes[0, 1].set(xlabel="Embedding dimension", ylabel="Variance")
    fig.suptitle(f"Clean validation embeddings: N={moments.n}, effective rank={effective_rank:.3g}")
    _save(fig, directory, "embeddings", figures)
    return {"status": "ok" if total > 0 else "zero_variance", "count": moments.n, "total_variance": total,
            "effective_rank": effective_rank, "maximum_empirical_rank": min(moments.n - 1, len(spectrum)),
            "near_constant_dimensions": int((variances <= 1.e-12).sum()), "variance_threshold": 1.e-12,
            "input": "unmasked validation; empty/nonfinite events excluded from covariance"}


def generate_pretraining_report(run_dir, *, output_dir=None, options=None, overwrite=False):
    """Stage diagnostics before publishing; preserve training outputs and CPU RNG."""
    from flowyforge.data_plugins.collide_v2.provenance import sha256_file
    from flowyforge.data_plugins.collide_v2.streaming import load_partition_manifest
    from flowyforge.model_plugins.foundation_encoder import encoder_contract
    from flowyforge.training.classification_run import _manifest_digest

    run = Path(run_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir else run / "pretraining_report"
    metrics = json.loads((run / "metrics.json").read_text())
    history = json.loads((run / "history.json").read_text())
    if metrics.get("kind") != "masked_pretraining" or metrics.get("version") != 1 or not history:
        raise ValueError("Expected masked_pretraining v1 metrics and a nonempty history.")
    if metrics.get("history") != history or metrics.get("test_used") is not False:
        raise ValueError("Inconsistent history/metrics or test exclusion not established.")
    best = metrics["best_epoch"]
    eligible = [e for e in history if e["validation"].get("loss") is not None and math.isfinite(e["validation"]["loss"])]
    if not eligible or min(eligible, key=lambda e: e["validation"]["loss"])["epoch"] != best:
        raise ValueError("Best epoch must be selected by minimum validation loss.")
    if (metrics["validation"] != next(e["validation"] for e in eligible if e["epoch"] == best)
            or metrics["constant_baseline"].get("split") != "train"):
        raise ValueError("Selected validation metrics or training-only constant are inconsistent.")
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    options = report_options({**config.get("reporting", {}), **(options or {})})
    if output.exists() and any(output.iterdir()):
        marker = output / "report.json"
        if not overwrite or not marker.exists() or json.loads(marker.read_text()).get("kind") != KIND:
            raise FileExistsError(f"Report output already exists or is not an owned report: {output}; choose another --output-dir or --overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"kind": KIND, "version": 1, "run_dir": str(run), "view": metrics["view"], "best_epoch": best,
              "split": "validation", "test_used": False, "options": options,
              "partition_sha256": metrics["partition_sha256"], "validation": metrics["validation"],
              "constant_baseline": metrics["constant_baseline"], "config": config,
              "sampling": "lowest SHA256(report-v1|seed|event_id) on validation; not a uniform sample of COLLIDE",
              "warnings": ["Training loss uses changing masks and online weights; validation masks are fixed.",
                           "Physical true values are reader-clipped GeV after expm1, not original unclipped values.",
                           "The physical constant is expm1(mean training log1p values), not mean physical PT/mass or the Huber-optimal constant.",
                           "Bias is mean residual; resolution is sample standard deviation. Relative residual divides by true > 0.",
                           "Target rows within an event are correlated; bin counts are not independent-event uncertainties.",
                           "Embedding covariance uses unmasked nonempty events; rank is limited by sample size. No collapse threshold is inferred."]}
    fingerprints = {name: sha256_file(run / name) for name in ("metrics.json", "history.json", "resolved_config.yaml")}
    with tempfile.TemporaryDirectory(prefix=".pretraining-report-", dir=output.parent) as temporary:
        directory, figures = Path(temporary), []
        _history_report(history, directory, figures)
        report["inference"] = {"status": "disabled"}
        if options["inference"]:
            path = run / "checkpoint.pt"
            if not path.exists():
                report["inference"] = {"status": "unavailable", "reason": "checkpoint.pt absent; history-only report"}
            else:
                fingerprints["checkpoint.pt"] = sha256_file(path)
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                if (checkpoint["best_epoch"] != best or checkpoint["history"] != history
                        or checkpoint["partition_sha256"] != metrics["partition_sha256"]
                        or checkpoint["constant_baseline"] != metrics["constant_baseline"]
                        or checkpoint["config"] != config):
                    raise ValueError("Checkpoint and saved metrics/configuration refer to different runs or epochs.")
                canonical = encoder_contract(config["model"], config["data"])
                if checkpoint["contract"] != canonical:
                    raise ValueError("Incompatible reconstruction input/architecture contract.")
                state, source = selected_model_state(checkpoint)
                if state is None:
                    report["inference"] = {"status": "unavailable", "reason": source}
                else:
                    # Also verify the independently exported downstream encoder when present.
                    if (run / "encoder.pt").exists():
                        fingerprints["encoder.pt"] = sha256_file(run / "encoder.pt")
                        exported = torch.load(run / "encoder.pt", weights_only=True, map_location="cpu")
                        expected = checkpoint["best_encoder_state_dict"]
                        if (exported.get("contract") != canonical or exported.get("provenance", {}).get("best_epoch") != best
                                or exported.get("state_dict", {}).keys() != expected.keys()
                                or any(not torch.equal(exported["state_dict"][k], v) for k, v in expected.items())):
                            raise ValueError("Exported encoder and selected full-model checkpoint are incompatible.")
                    manifest = json.loads((run / "dataset_manifest.json").read_text())
                    partitions = load_partition_manifest(run / "partitions.json")
                    if (_manifest_digest(manifest) != checkpoint["dataset_sha256"]
                            or partitions["sha256"] != checkpoint["partition_sha256"]
                            or manifest.get("partition_sha256") != partitions["sha256"] or manifest["reader"] != canonical["reader"]):
                        raise ValueError("Diagnostic source/partition manifest differs from the saved checkpoint.")
                    fingerprints.update({name: sha256_file(run / name) for name in ("dataset_manifest.json", "partitions.json")})
                    # CPU inference avoids device-dependent stochastic layers; restore caller RNG and threads.
                    threads = torch.get_num_threads()
                    try:
                        torch.set_num_threads(config["training"]["cpu_threads"])
                        with torch.random.fork_rng(devices=[]):
                            inferred, ranges, moments = _infer(checkpoint, state, manifest, partitions, options, directory)
                    finally:
                        torch.set_num_threads(threads)
                    report["inference"] = {"status": "ok", "state_source": source, "epoch": best, "device": "cpu", **inferred}
                    keys = [f"{c}/{f}" for c, fs in config["pretraining"]["targets"].items() for f in fs]
                    report["features"] = _prediction_plots(directory, ranges, options, figures, keys)
                    report["embeddings"] = _embedding_report(moments, directory, figures)
        for name, digest in fingerprints.items():
            if sha256_file(run / name) != digest:
                raise ValueError(f"Training artifact changed during report generation: {name}")
        report["fingerprints"] = fingerprints
        report["software"] = {"torch": str(torch.__version__), "numpy": np.__version__, "pyarrow": pa.__version__}
        report["figures"] = figures
        report["html"] = str(output / "pretraining_report.html")
        (directory / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report["artifacts"] = {p.name: {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
                               for p in sorted(directory.iterdir())}
        _json(directory / "report.json", report)
        body = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Masked pretraining diagnostics</title>",
                "<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:1em}img{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}figure{margin:2em 0}</style>",
                f"<h1>Masked pretraining — {html.escape(report['view'])}</h1>",
                f"<p>Selected epoch: {best}. Prediction split: validation only. Inference: {html.escape(report['inference']['status'])}.</p>",
                "<h2>Metrics and limitations</h2><pre>" + html.escape(json.dumps(_clean({k: v for k, v in report.items() if k not in {'config', 'figures'}}), indent=2)) + "</pre>"]
        for name in figures:
            body.append(f"<figure><img src='{name}.png' alt='{name}'><figcaption>{name} · <a href='{name}.pdf'>PDF</a></figcaption></figure>")
        body.append("<h2>Configuration</h2><pre>" + html.escape(yaml.safe_dump(config, sort_keys=False)) + "</pre><h2>Artifacts</h2><ul>")
        for artifact in sorted(directory.iterdir()):
            body.append(f"<li><a href='{artifact.name}'>{artifact.name}</a></li>")
        body.append("</ul></html>")
        (directory / "pretraining_report.html").write_text("\n".join(body), encoding="utf-8")
        output.mkdir(parents=True, exist_ok=True)
        # Replace only generated report files, never remove datasets/checkpoints or unrelated outputs.
        for artifact in directory.iterdir():
            artifact.replace(output / artifact.name)
    return report
