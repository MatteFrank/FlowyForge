"""Shareable classification reports from streaming aggregate statistics."""

from __future__ import annotations

import base64
import csv
import html
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flowyforge.data_plugins.collide_v2.provenance import sha256_file


def compute_binary_metrics(scores, labels, *, signal_label: int, background_acceptance=0.01,
                           min_background_events=10) -> dict[str, Any]:
    """Exact unweighted ROC, grouping ties before counting accepted events.

    AUC integrates the empirical ROC (tied signal/background pairs count 1/2).
    The operating point uses score >= threshold, never splits a tie or
    interpolates an efficiency at an unattainable background acceptance.
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if (scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1)) or signal_label not in (0, 1)
        or not np.isin(labels, [0, 1]).all()):
        raise ValueError("Binary metrics require finite probability scores, labels 0/1 and aligned one-dimensional arrays.")
    if not math.isfinite(background_acceptance) or not 0 < background_acceptance <= 1:
        raise ValueError("background_acceptance must be in (0, 1].")
    if type(min_background_events) is not int or min_background_events < 1:
        raise ValueError("min_background_events must be a positive integer.")
    signal = labels == signal_label
    ns, nb = int(signal.sum()), int((~signal).sum())
    point = {"target_background_acceptance": background_acceptance,
             "min_background_events": min_background_events,
             "background_events_at_target": nb * background_acceptance,
             "status": "missing_class", "signal_efficiency": None, "threshold": None,
             "accepted_signal": None, "accepted_background": None, "achieved_background_acceptance": None}
    result = {"signal_label": signal_label, "num_signal": ns, "num_background": nb,
              "roc_auc": None, "roc": [], "background_fraction_resolution": 1 / nb if nb else None,
              "operating_point": point, "status": "missing_class"}
    if not ns or not nb:
        return result
    order = np.argsort(-scores, kind="stable")
    ordered, positives = scores[order], signal[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(scores) - 1]
    tp = np.r_[0, np.cumsum(positives, dtype=np.int64)[ends]]
    fp = np.r_[0, ends + 1 - tp[1:]]
    thresholds = [None, *ordered[ends].tolist()]  # None means reject all, not score >= 0.
    tpr, fpr = tp / ns, fp / nb
    result.update(status="ok", roc_auc=float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2)),
                  roc=[{"threshold": threshold, "false_positive_rate": float(x), "true_positive_rate": float(y),
                        "accepted_signal": int(s), "accepted_background": int(b)}
                       for threshold, x, y, s, b in zip(thresholds, fpr, tpr, tp, fp)])
    budget = math.floor(np.nextafter(nb * background_acceptance, math.inf))
    if budget < min_background_events:
        point["status"] = "insufficient_background_resolution"
        return result
    index = int(np.flatnonzero(fp <= budget)[-1])
    point.update(status="ok", signal_efficiency=float(tpr[index]), threshold=thresholds[index],
                 accepted_signal=int(tp[index]), accepted_background=int(fp[index]),
                 achieved_background_acceptance=float(fpr[index]), tie_limited=bool(fp[index] < budget))
    return result


class EvaluationScoreWriter:
    """Bounded scalar buffer; write immutable Parquet shards and a completion manifest."""

    def __init__(self, output_dir, split, *, context, shard_rows=10000):
        if split not in {"validation", "test"} or type(shard_rows) is not int or shard_rows < 1:
            raise ValueError("Scores require validation/test and a positive shard_rows.")
        self.directory = Path(output_dir) / "scores" / split
        self.directory.mkdir(parents=True, exist_ok=False)
        self.split, self.context, self.shard_rows = split, dict(context), shard_rows
        self.rows, self.shards, self.count = [], [], 0
        self.schema = pa.schema([("event_id", pa.string()), ("label", pa.int64()),
                                 ("signal_score", pa.float64()), ("split", pa.string())],
                                metadata={b"flowyforge_scores": json.dumps(self.context, sort_keys=True).encode()})

    def append(self, event_metadata, labels, scores):
        if not len(event_metadata) == len(labels) == len(scores):
            raise ValueError("Score batch metadata/labels/scores differ in length.")
        for metadata, label, score in zip(event_metadata, labels, scores):
            identity = metadata.get("event_id")
            if not isinstance(identity, str) or not identity.startswith("origin-v1:"):
                raise ValueError("Evaluation scores require stable event identities from shared partitions.")
            if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid evaluation label or signal score.")
            self.rows.append({"event_id": identity, "label": int(label), "signal_score": float(score), "split": self.split})
            if len(self.rows) == self.shard_rows:
                self._flush()

    def _flush(self):
        if not self.rows:
            return
        path = self.directory / f"part-{len(self.shards):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows, schema=self.schema), path, compression="zstd")
        self.shards.append({"path": path.name, "num_rows": len(self.rows), "sha256": sha256_file(path)})
        self.count += len(self.rows)
        self.rows.clear()

    def finish(self):
        self._flush()
        manifest = {"version": 1, "split": self.split, "context": self.context,
                    "num_rows": self.count, "shards": self.shards}
        with (self.directory / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
        return str(Path("scores") / self.split / "manifest.json")


def read_evaluation_scores(path):
    """Load scalar scores/IDs only, checking every shard before use. No event objects."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("split") not in {"validation", "test"}:
        raise ValueError("Unsupported score manifest version/split.")
    parts, seen_paths = [], set()
    for info in manifest["shards"]:
        name = info["path"]
        if Path(name).name != name or name in seen_paths:
            raise ValueError("Invalid or duplicate score shard path.")
        seen_paths.add(name)
        shard = path.parent / name
        if sha256_file(shard) != info["sha256"]:
            raise ValueError(f"Score shard content changed: {shard}")
        table = pq.read_table(shard)
        if (table.column_names != ["event_id", "label", "signal_score", "split"]
            or len(table) != info["num_rows"]
            or json.loads((table.schema.metadata or {}).get(b"flowyforge_scores", b"{}")) != manifest["context"]):
            raise ValueError("Score shard schema/count/context differs from manifest.")
        if any(table[name].null_count for name in table.column_names):
            raise ValueError("Null values in evaluation scores.")
        if set(table["split"].to_pylist()) != {manifest["split"]}:
            raise ValueError("Score shard contains a different split.")
        parts.append(table)
    if not parts or sum(map(len, parts)) != manifest["num_rows"]:
        raise ValueError("Empty or incomplete evaluation score shards.")
    table = pa.concat_tables(parts)
    ids = table["event_id"].to_pylist()
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate evaluation event identities.")
    labels, scores = table["label"].to_numpy(), table["signal_score"].to_numpy()
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid evaluation labels/scores.")
    return manifest, ids, labels, scores


def _load_comparison_run(directory, *, require_scratch=True):
    """Shared manifest/identity/score checks for view and downstream comparisons."""
    from flowyforge.data_plugins.collide_v2.streaming import load_partition_manifest

    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if not metrics.get("config", {}).get("evaluation", {}).get("signal_process"):
        raise ValueError("Comparison requires binary evaluation with shared partitions; use the supervised reference config.")
    manifest = json.loads((directory / "dataset_manifest.json").read_text(encoding="utf-8"))
    partitions = load_partition_manifest(directory / "partitions.json")
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, allow_nan=False).encode()).hexdigest()
    if (manifest.get("version") != 2 or digest != metrics.get("dataset_manifest_sha256") or
        partitions["sha256"] != manifest.get("partition_sha256") or partitions["sha256"] != metrics.get("partition_sha256") or
        manifest["label_map"] != metrics["label_map"] or manifest["label_map"] != partitions["label_map"] or
        manifest["reader"]["view"] != metrics["view"]):
        raise ValueError("Run dataset/partition manifest differs from evaluated metrics.")
    if require_scratch and (metrics.get("initialization") != "random_from_scratch" or not metrics.get("initial_state_sha256")):
        raise ValueError("Comparison requires a recorded random initialization from scratch.")
    if metrics.get("checkpoint_selection") != "minimum validation loss":
        raise ValueError("Comparison requires validation-based checkpoint selection.")
    expected = {split: {} for split in ("train", "validation", "test")}
    for process in partitions["processes"]:
        for event in process["events"]:
            if event["split"] is not None:
                expected[event["split"]][event["id"]] = process["label"]
    bound = {split: {} for split in expected}
    for process in manifest["processes"]:
        keys = [key for info in process["files"] for key in info["event_ids"]]
        for split in expected:
            for index in process["split_indices"][split]:
                key = keys[index]
                if key in bound[split]:
                    raise ValueError("Duplicate event in run split binding.")
                bound[split][key] = process["label"]
    if bound != expected:
        raise ValueError("Run events/splits differ from the saved partition memberships.")
    arrays = {}
    for split in ("validation", "test"):
        # Fixed layout prevents a metrics file from redirecting score reads.
        score_manifest, ids, labels, scores = read_evaluation_scores(directory / "scores" / split / "manifest.json")
        context = score_manifest["context"]
        if (dict(zip(ids, labels.tolist())) != expected[split] or
            any(context.get(key) != metrics.get(key) for key in
                ("partition_sha256", "dataset_manifest_sha256", "checkpoint_sha256", "view", "label_map"))):
            raise ValueError(f"Scored {split} events/labels/context differ from saved partitions/run.")
        evaluation = metrics["config"]["evaluation"]
        if (context["signal_process"] != evaluation["signal_process"] or
            context["signal_label"] != metrics["label_map"].get(evaluation["signal_process"])):
            raise ValueError("Score signal definition differs from run configuration.")
        arrays[split] = (labels, scores)
    return metrics, arrays


def _comparison_rows(metrics, arrays):
    rows = []
    settings = metrics["config"]["evaluation"]
    for split, (labels, scores) in arrays.items():
        binary = compute_binary_metrics(scores, labels, signal_label=metrics["label_map"][settings["signal_process"]],
                    background_acceptance=settings["background_acceptance"], min_background_events=settings["min_background_events"])
        recorded = metrics[split]["binary"]
        if any(recorded.get(key) != value for key, value in binary.items()):
            raise ValueError("Recorded binary metrics differ from the persisted evaluation scores.")
        point = binary["operating_point"]
        rows.append({"view": metrics["view"], "split": split, "signal_process": settings["signal_process"],
                     "best_epoch": metrics["best_epoch"], "accuracy": metrics[split]["accuracy"],
                     "macro_f1": metrics[split]["macro_f1"], "roc_auc": binary["roc_auc"],
                     "num_signal": binary["num_signal"], "num_background": binary["num_background"],
                     **{key: point[key] for key in ("target_background_acceptance", "signal_efficiency",
                                                   "achieved_background_acceptance", "accepted_background", "status")}})
    return rows


def compare_classification_runs(run_dirs, output_dir):
    """Compare two views only after matching partitions, protocol and scored IDs.

    Accept training or standalone evaluation directories; no raw data/model load.
    Exact ROC uses scalar arrays, not physical event objects or model tensors.
    """
    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(paths) != 2 or paths[0] == paths[1]:
        raise ValueError("Comparison requires two distinct run directories.")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Comparison output must be new or empty; refusing to overwrite existing artifacts.")
    runs, scored = [], []
    for directory in paths:
        metrics, arrays = _load_comparison_run(directory)
        runs.append(metrics)
        scored.append(arrays)
    left, right = runs
    if {left["view"], right["view"]} != {"FullReco", "L1T"}:
        raise ValueError("Comparison requires one FullReco run and one L1T run.")
    for key in ("partition_sha256", "label_map", "training_protocol", "initial_state_sha256"):
        if left.get(key) is None or left[key] != right.get(key):
            raise ValueError(f"Runs are not comparable: {key} differs.")
    for key in ("signal_process", "background_acceptance", "min_background_events"):
        if left["config"]["evaluation"][key] != right["config"]["evaluation"][key]:
            raise ValueError(f"Runs are not comparable: evaluation.{key} differs.")
    # All event and protocol checks have passed before comparing any metrics.
    rows = [row for metrics, arrays in zip(runs, scored) for row in _comparison_rows(metrics, arrays)]
    comparison = {"version": 1, "run_dirs": list(map(str, paths)), "partition_sha256": left["partition_sha256"],
                  "verified": ["all selected event identities and split memberships", "score identities and labels",
                               "same architecture, reader settings except view, training protocol and initial weights",
                               "binary metrics recomputed from immutable score shards"], "results": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output / "comparison.csv", rows, list(rows[0]))
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>FlowyForge view comparison</title>'
                '<style>body{font:15px system-ui;margin:30px}.table-wrap{overflow:auto}table{border-collapse:collapse}'
                'td,th{padding:8px;border:1px solid #ccc;text-align:left}</style><h1>FullReco / L1T comparison</h1>'
                '<p>Same events, partitions, training protocol and initial weights verified. '
                'Unweighted empirical metrics; test results must not guide hyperparameter selection.</p>'
                + _table(rows, list(rows[0])) + '</html>')
    (output / "comparison.html").write_text(document, encoding="utf-8")
    return comparison


def _verify_downstream_checkpoint(directory, metrics):
    """Cross-check recorded protocol, freezing and fingerprints against saved weights."""
    import torch
    from flowyforge.model_plugins.foundation_encoder import encoder_contract
    from flowyforge.training.classification_run import _make_model, _parameter_counts, _initial_state_digest, _training_protocol
    from flowyforge.data_plugins.collide_v2.streaming import _reader_config

    checkpoint_path = directory / "checkpoint.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = Path(metrics.get("checkpoint_path", ""))
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != metrics.get("checkpoint_sha256"):
        raise ValueError("Downstream checkpoint missing or fingerprint differs from evaluated scores.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2 or not checkpoint.get("config", {}).get("downstream"):
        raise ValueError("Transfer comparison requires explicit downstream runs (checkpoint format_version=2).")
    config = checkpoint["config"]
    downstream = metrics.get("downstream", {})
    mode = config["downstream"]["mode"]
    if mode not in {"scratch", "linear_probe", "finetune"}:
        raise ValueError("Unsupported downstream mode.")
    contract = encoder_contract(config["model"], config["data"])
    for key in ("downstream", "input_contract", "training_protocol", "initial_state_sha256", "initialization", "label_map", "best_epoch", "best_validation_loss"):
        if checkpoint.get(key) != metrics.get(key):
            raise ValueError(f"Downstream metrics/checkpoint mismatch: {key}.")
    if (downstream.get("mode") != mode or downstream.get("head_type") != "linear"
        or downstream.get("version") != 1 or metrics["input_contract"] != contract
        or metrics["training_protocol"] != _training_protocol(config)
        or encoder_contract(metrics["config"]["model"], metrics["config"]["data"]) != contract
        or metrics["config"].get("downstream") != config["downstream"]
        or metrics["view"] != contract["reader"]["view"]):
        raise ValueError("Downstream input contract, head or training protocol is inconsistent.")
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    if manifest["reader"] != _reader_config(**contract["reader"]):
        raise ValueError("Downstream reader binding differs from the encoder contract.")
    for key in ("signal_process", "background_acceptance", "min_background_events"):
        if config["evaluation"][key] != metrics["config"]["evaluation"][key]:
            raise ValueError(f"Downstream evaluation.{key} differs from checkpoint protocol.")
    frozen = mode == "linear_probe"
    if downstream.get("freezing") != ("encoder_frozen_eval" if frozen else "none"):
        raise ValueError("Incorrect downstream freezing policy.")
    with torch.random.fork_rng(devices=[]):
        model = _make_model(config, metrics["label_map"], initialize_encoder=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if (downstream.get("parameters") != _parameter_counts(model)
        or downstream.get("selected_encoder_sha256") != _initial_state_digest(model.encoder)
        or downstream.get("selected_head_sha256") != _initial_state_digest(model.head)):
        raise ValueError("Downstream parameter counts or selected weight fingerprints are inconsistent.")
    if frozen and downstream["initial_encoder_sha256"] != downstream["selected_encoder_sha256"]:
        raise ValueError("Linear probe modified its frozen encoder.")
    source = downstream.get("encoder_source")
    if mode == "scratch":
        if source is not None or metrics["initialization"] != "random_from_scratch":
            raise ValueError("Scratch must use random initialization without an encoder source.")
    else:
        if (not source or not source.get("sha256") or metrics["initialization"] != "pretrained_encoder_new_head"
            or source.get("provenance", {}).get("partition_sha256") != metrics["partition_sha256"]
            or source["provenance"].get("test_used") is not False
            or source["provenance"].get("pretraining_split") != "train"
            or source["provenance"].get("checkpoint_selection_split") != "validation"):
            raise ValueError("Incompatible pretraining provenance or test exclusion.")
    history = metrics.get("history") or []
    if (len(history) != config["training"]["epochs"]
        or [item["epoch"] for item in history] != list(range(1, len(history) + 1))
        or any(not math.isfinite(item["validation"]["loss"]) for item in history)):
        raise ValueError("Incomplete or invalid downstream learning history.")
    best = min(history, key=lambda item: item["validation"]["loss"])
    if best["epoch"] != metrics["best_epoch"] or best["validation"]["loss"] != metrics["best_validation_loss"]:
        raise ValueError("Checkpoint selection differs from minimum validation loss.")
    for split in ("validation", "test"):
        recomputed = compute_confusion_metrics(metrics[split]["confusion_matrix"], metrics["label_map"])
        for key in ("accuracy", "macro_f1", "num_events"):
            if recomputed[key] != metrics[split][key]:
                raise ValueError(f"Recorded {split} {key} differs from confusion counts.")


def compare_downstream_runs(run_dirs, output_dir):
    """Compare all three modes for ONE view, after verifying experimental conditions."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    from flowyforge.training.classification_run import _performance

    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    if len(paths) != 3 or len(set(paths)) != 3:
        raise ValueError("Transfer comparison requires three distinct runs: scratch, linear_probe, finetune.")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Comparison output must be new or empty; refusing to overwrite existing artifacts.")
    runs, scored = {}, {}
    for directory in paths:
        metrics, arrays = _load_comparison_run(directory, require_scratch=False)
        _verify_downstream_checkpoint(directory, metrics)
        mode = metrics["downstream"]["mode"]
        if mode in runs:
            raise ValueError(f"Duplicate downstream mode: {mode}.")
        runs[mode], scored[mode] = metrics, arrays
    modes = ("scratch", "linear_probe", "finetune")
    if set(runs) != set(modes):
        raise ValueError("Transfer comparison requires scratch, linear_probe and finetune.")
    reference = runs["scratch"]
    for metrics in runs.values():
        for key in ("view", "partition_sha256", "label_map", "input_contract", "training_protocol"):
            if metrics[key] != reference[key]:
                raise ValueError(f"Downstream runs are not comparable: {key} differs.")
        for key in ("signal_process", "background_acceptance", "min_background_events"):
            if metrics["config"]["evaluation"][key] != reference["config"]["evaluation"][key]:
                raise ValueError(f"Downstream runs are not comparable: evaluation.{key} differs.")
        if metrics["downstream"]["initial_head_sha256"] != reference["downstream"]["initial_head_sha256"]:
            raise ValueError("Downstream runs must start with the same fresh linear head.")
    probe, tuned = (runs[mode]["downstream"] for mode in ("linear_probe", "finetune"))
    if (probe["encoder_source"]["sha256"] != tuned["encoder_source"]["sha256"]
        or probe["initial_encoder_sha256"] != tuned["initial_encoder_sha256"]
        or probe["encoder_source"]["provenance"] != tuned["encoder_source"]["provenance"]):
        raise ValueError("Linear probe and fine-tuning must start from the same pretrained encoder.")
    rows, learning = [], []
    for mode in modes:
        metrics = runs[mode]
        performance = _performance(metrics["history"])
        if performance != metrics.get("performance"):
            raise ValueError("Recorded throughput/duration differ from learning history.")
        for row in _comparison_rows(metrics, scored[mode]):
            rows.append({"mode": mode, **row, "validation_loss": metrics["best_validation_loss"],
                         "total_parameters": metrics["downstream"]["parameters"]["total"],
                         "trainable_parameters": metrics["downstream"]["parameters"]["trainable"],
                         "train_duration_seconds": performance["train_duration_seconds"],
                         "train_events_per_second": performance["train_events_per_second"],
                         "evaluation_duration_seconds": metrics[row["split"]]["duration_seconds"],
                         "evaluation_events_per_second": metrics[row["split"]]["events_per_second"]})
        learning.extend({"mode": mode, "epoch": item["epoch"], "train_loss": item["train_loss"],
                         "validation_loss": item["validation"]["loss"], "train_accuracy": item["train_accuracy"],
                         "validation_accuracy": item["validation"]["accuracy"]} for item in metrics["history"])
    # No artifacts are created before all consistency checks, including metrics, pass.
    comparison = {"version": 1, "kind": "downstream_transfer_comparison", "run_dirs": list(map(str, paths)),
                  "view": reference["view"], "partition_sha256": reference["partition_sha256"],
                  "training_protocol": reference["training_protocol"], "input_contract": reference["input_contract"],
                  "verified": ["identical selected event IDs, split memberships, labels and task",
                               "identical input contract, view, signal and optimization protocol",
                               "same fresh linear head; same pretrained encoder for linear_probe/finetune",
                               "checkpoint fingerprints, parameter counts and frozen encoder unchanged",
                               "minimum validation loss selection; binary metrics recomputed from score shards"],
                  "intentional_differences": ["encoder initialization", "encoder freezing and trainable parameter count"],
                  "encoder_sources": {mode: runs[mode]["downstream"]["encoder_source"] for mode in modes},
                  "results": rows}
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_csv(output / "comparison.csv", rows, list(rows[0]))
    _write_csv(output / "learning_curves.csv", learning, list(learning[0]))
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    images = []
    def save(fig, stem):
        fig.savefig(plots / f"{stem}.png", dpi=160, facecolor="white")
        fig.savefig(plots / f"{stem}.pdf", facecolor="white")
        images.append((stem, base64.b64encode((plots / f"{stem}.png").read_bytes()).decode()))
        fig.clear()
    colors = {"scratch": "#2463a6", "linear_probe": "#df8b29", "finetune": "#278575"}
    for split in ("validation", "test"):
        fig = Figure(figsize=(7.5, 5.5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        for mode in modes:
            binary = runs[mode][split]["binary"]
            roc = binary["roc"]
            if roc:
                ax.plot([p["false_positive_rate"] for p in roc], [p["true_positive_rate"] for p in roc],
                        color=colors[mode], label=f"{mode}: AUC={binary['roc_auc']:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=.6)
        ax.set(xlabel="Accepted background fraction (FPR)", ylabel="Signal efficiency (TPR)",
               title=f"{reference['view']} · {split} ROC", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
        save(fig, f"roc_{split}")
    fig = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(fig)
    for ax, metric in zip(fig.subplots(1, 2), ("loss", "accuracy")):
        for mode in modes:
            points = [row for row in learning if row["mode"] == mode]
            for split, style in (("train", "--"), ("validation", "-")):
                ax.plot([p["epoch"] for p in points], [p[f"{split}_{metric}"] for p in points], style,
                        marker="o", markersize=3, color=colors[mode], label=f"{mode} {split}")
        ax.set(xlabel="Epoch", ylabel=metric, title=f"{reference['view']} · {metric}")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    save(fig, "learning_curves")
    document = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>FlowyForge transfer comparison</title>'
                '<style>body{font:15px system-ui;margin:30px}.table-wrap{overflow:auto}table{border-collapse:collapse}'
                'td,th{padding:8px;border:1px solid #ccc;text-align:left}img{max-width:100%}</style>'
                '<h1>Scratch / linear probe / fine-tuning</h1>'
                '<p>Same events, splits, input contract, task, signal and optimization protocol verified. '
                'All heads are strictly linear. Encoder initialization and freezing intentionally differ. '
                'Test is excluded from checkpoint selection and must not guide tuning. '
                'Unweighted single-seed results do not establish statistical significance. '
                'Efficiency is an empirical ROC operating point, not a calibrated deployment threshold. '
                'Unavailable efficiency means insufficient background resolution. '
                'Throughput includes data I/O and excludes pretraining; compare timing only on comparable hardware.</p>'
                + _table(rows, list(rows[0]))
                + ''.join(f'<h2>{html.escape(stem)}</h2><img src="data:image/png;base64,{data}" alt="{stem}">' for stem, data in images)
                + '</html>')
    (output / "comparison.html").write_text(document, encoding="utf-8")
    return comparison


def compute_confusion_metrics(matrix: Any, label_map: dict[str, int]) -> dict[str, Any]:
    """Rows=true, columns=predicted; undefined per-class scores are zero.

    Balanced accuracy averages recall over classes with true support; macro F1
    includes every configured class. Event counts and scores are unweighted.
    """
    counts = np.asarray(matrix)
    size = len(label_map)
    if set(label_map.values()) != set(range(size)):
        raise ValueError("label_map must contain contiguous class IDs starting at zero")
    if counts.shape != (size, size) or size == 0:
        raise ValueError("confusion matrix must be square and match label_map")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("confusion matrix must contain finite nonnegative counts")
    if np.any(counts != np.floor(counts)):
        raise ValueError("confusion matrix must contain integer event counts")
    counts = counts.astype(np.int64)
    support, predicted, correct = counts.sum(axis=1), counts.sum(axis=0), np.diag(counts)
    precision = np.divide(correct, predicted, out=np.zeros(size), where=predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(size), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(size), where=(precision + recall) > 0)
    names = {class_id: process for process, class_id in label_map.items()}
    total = int(support.sum())
    return {
        "num_events": total,
        "accuracy": float(correct.sum() / total) if total else 0.0,
        "balanced_accuracy": float(recall[support > 0].mean()) if total else 0.0,
        "macro_f1": float(f1.mean()), "confusion_matrix": counts.tolist(),
        "per_class": [
            {"class_id": index, "process": names[index], "support": int(support[index]),
             "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])}
            for index in range(size)
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if value != 0 and abs(value) < 1e-4:
            return f"{value:.3g}"
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "".join(f"<th>{html.escape(key.replace('_', ' ').title())}</th>" for key in columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_format(row.get(key)))}</td>" for key in columns) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _flatten(value: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [row for key, item in value.items()
                for row in _flatten(item, f"{prefix}.{key}" if prefix else str(key))]
    return [{"quantity": prefix, "value": value}]


def write_classification_report(metrics: dict[str, Any], output_dir: str | Path) -> Path:
    """Write CSV, PNG/PDF figures and self-contained report.html; return its path.

    Optional inputs: history, train/validation/test, split_class_counts,
    diagnostics, preflight, warnings, config. No event predictions are required.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    import matplotlib as mpl

    output = Path(output_dir)
    plots_dir, tables_dir = output / "plots", output / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    label_map = metrics.get("label_map", {})
    names = [name for name, _ in sorted(label_map.items(), key=lambda pair: pair[1])]
    figures: list[tuple[str, Path]] = []
    tables: list[tuple[str, list[dict[str, Any]], list[str]]] = []
    colors = ["#2463a6", "#df8b29", "#278575"]

    def figure(width: float = 10, height: float = 5) -> Figure:
        fig = Figure(figsize=(min(width, 24), min(height, 24)), layout="constrained")
        FigureCanvasAgg(fig)
        return fig

    def save(fig: Figure, stem: str, title: str) -> None:
        fig.savefig(plots_dir / f"{stem}.png", dpi=170, facecolor="white")
        fig.savefig(plots_dir / f"{stem}.pdf", facecolor="white")
        figures.append((title, plots_dir / f"{stem}.png"))
        fig.clear()

    def table(stem: str, title: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
        _write_csv(tables_dir / f"{stem}.csv", rows, columns)
        tables.append((title, rows, columns))

    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 13,
             "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
             "axes.titleweight": "bold", "figure.titlesize": 16, "savefig.bbox": "tight"}
    with mpl.rc_context(style):
        history = metrics.get("history") or []
        if history:
            rows = [{"epoch": item.get("epoch", index + 1),
                     "train_loss": item.get("train_loss"), "train_accuracy": item.get("train_accuracy"),
                     "validation_loss": (item.get("validation") or {}).get("loss"),
                     "validation_accuracy": (item.get("validation") or {}).get("accuracy"),
                     "duration_seconds": item.get("duration_seconds")}
                    for index, item in enumerate(history)]
            table("history", "Learning history (training metrics are online)", rows, list(rows[0]))
            fig = figure(height=4.3)
            for ax, score, title in zip(fig.subplots(1, 2), ("loss", "accuracy"), ("Cross-entropy loss", "Accuracy")):
                for prefix, color in zip(("train", "validation"), colors):
                    points = [(row["epoch"], row[f"{prefix}_{score}"]) for row in rows if row[f"{prefix}_{score}"] is not None]
                    if points:
                        label = "Train (online)" if prefix == "train" else "Validation"
                        ax.plot(*zip(*points), label=label, color=color, linewidth=2, marker="o", markersize=3)
                if metrics.get("best_epoch") is not None:
                    ax.axvline(metrics["best_epoch"], color="#8795a5", linestyle="--", linewidth=1, label="Selected epoch")
                ax.set(xlabel="Epoch", ylabel=title, title=title)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_xlim(min(row["epoch"] for row in rows) - 0.5, max(row["epoch"] for row in rows) + 0.5)
                ax.grid(alpha=0.18)
                if score == "accuracy":
                    ax.set_ylim(0, 1.02)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(frameon=False)
            save(fig, "learning_curves", "Learning curves")

        summary = []
        for split in ("train", "validation", "test"):
            result = metrics.get(split)
            if not isinstance(result, dict):
                continue
            result = dict(result)
            matrix = result.get("confusion_matrix")
            if matrix is not None and label_map:
                result.update(compute_confusion_metrics(matrix, label_map))
            summary.append({"split": split, **{key: result.get(key) for key in
                           ("num_events", "loss", "accuracy", "balanced_accuracy", "macro_f1")}})
            binary = result.get("binary")
            if binary:
                point = binary["operating_point"]
                row = {"signal_process": binary["signal_process"], "signal_label": binary["signal_label"],
                       "background_process": binary["background_process"], "num_signal": binary["num_signal"],
                       "num_background": binary["num_background"], "roc_auc": binary["roc_auc"],
                       "roc_status": binary["status"], "background_fraction_resolution": binary["background_fraction_resolution"],
                       **point}
                table(f"binary_{split}", f"{split.title()} binary signal/background metrics", [row], list(row))
                roc = binary["roc"]
                _write_csv(tables_dir / f"roc_{split}.csv", roc,
                           ["threshold", "false_positive_rate", "true_positive_rate", "accepted_signal", "accepted_background"])
                if roc:
                    fig = figure(width=7, height=5.5)
                    ax = fig.subplots()
                    ax.plot([p["false_positive_rate"] for p in roc], [p["true_positive_rate"] for p in roc],
                            color=colors[0], label=f"AUC = {binary['roc_auc']:.4f}")
                    ax.plot([0, 1], [0, 1], color="#8795a5", linestyle="--", label="Random ranking")
                    if point["signal_efficiency"] is not None:
                        ax.scatter([point["achieved_background_acceptance"]], [point["signal_efficiency"]],
                                   color=colors[1], zorder=3, label="Attainable operating point")
                    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Accepted background fraction (FPR)",
                           ylabel="Signal efficiency (TPR)",
                           title=f"{metrics.get('view', '')} {split.title()} ROC\nSignal: {binary['signal_process']}")
                    ax.legend(frameon=False)
                    ax.grid(alpha=0.18)
                    save(fig, f"roc_{split}", f"{split.title()} ROC · signal: {binary['signal_process']}")
            per_class = result.get("per_class") or []
            if per_class:
                table(f"per_class_{split}", f"{split.title()} per-class metrics", per_class,
                      ["class_id", "process", "support", "precision", "recall", "f1"])
                fig = figure(width=max(8, len(per_class) * 0.65), height=4.7)
                ax = fig.subplots()
                positions = np.arange(len(per_class))
                for offset, (score, color) in enumerate(zip(("precision", "recall", "f1"), colors)):
                    ax.bar(positions + (offset - 1) * 0.24, [row[score] for row in per_class], width=0.24, color=color, label=score.title())
                ax.set(xticks=positions, xticklabels=[row["process"] for row in per_class], ylim=(0, 1.06), ylabel="Score", title=f"{split.title()} · per-class performance")
                ax.tick_params(axis="x", labelrotation=35)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
                ax.legend(frameon=False, ncol=3, loc="upper right")
                ax.grid(axis="y", alpha=0.18)
                ax.set_axisbelow(True)
                save(fig, f"per_class_{split}", f"{split.title()} per-class performance")
            if matrix is not None and label_map:
                counts = np.asarray(matrix, dtype=np.int64)
                totals = counts.sum(axis=1, keepdims=True)
                normalized = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
                confusion_rows = [{"true_process": true_name, "predicted_process": predicted_name,
                                   "count": int(counts[row, col]), "row_fraction": float(normalized[row, col])}
                                  for row, true_name in enumerate(names) for col, predicted_name in enumerate(names)]
                _write_csv(tables_dir / f"confusion_{split}.csv", confusion_rows,
                           ["true_process", "predicted_process", "count", "row_fraction"])
                for suffix, values, title in (("counts", counts, "Event counts"), ("normalized", normalized, "Fraction of true class")):
                    extent = max(6.5, len(names) * 0.55)
                    fig = figure(width=extent + 1, height=extent)
                    ax = fig.subplots()
                    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if suffix == "normalized" else None)
                    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
                           xlabel="Predicted process", ylabel="True process", title=f"{split.title()} · {title.lower()}")
                    ax.tick_params(axis="x", labelrotation=45)
                    for tick in ax.get_xticklabels():
                        tick.set_ha("right")
                    if len(names) <= 20:
                        threshold = float(values.max()) * 0.55
                        for row in range(len(names)):
                            for col in range(len(names)):
                                value = values[row, col]
                                ax.text(col, row, f"{value:.2f}" if suffix == "normalized" else str(value),
                                        ha="center", va="center", fontsize=8, color="white" if value > threshold else "#172b40")
                    fig.colorbar(im, ax=ax, label=title, shrink=0.8)
                    save(fig, f"confusion_{split}_{suffix}", f"{split.title()} confusion matrix · {title.lower()}")
        if summary:
            table("summary", "Evaluation summary (selected checkpoint)", summary,
                  ["split", "num_events", "loss", "accuracy", "balanced_accuracy", "macro_f1"])

        split_counts = metrics.get("split_class_counts") or {}
        if split_counts:
            rows = []
            for split, counts in split_counts.items():
                for index, name in enumerate(names):
                    count = counts.get(name, 0)
                    rows.append({"split": split, "class_id": index, "process": name, "num_events": int(count)})
            table("split_class_counts", "Selected events by split and process", rows, ["split", "class_id", "process", "num_events"])
            if names:
                fig = figure(width=max(8, len(names) * 0.65), height=4.8)
                ax = fig.subplots()
                bottom = np.zeros(len(names))
                for index, split in enumerate(split_counts):
                    values = np.array([row["num_events"] for row in rows if row["split"] == split])
                    ax.bar(names, values, bottom=bottom, label=split.title(), color=colors[index % len(colors)])
                    bottom += values
                ax.set(ylabel="Selected events", title="Class composition of selected data")
                ax.tick_params(axis="x", labelrotation=35)
                for tick in ax.get_xticklabels():
                    tick.set_ha("right")
                ax.legend(frameon=False)
                ax.grid(axis="y", alpha=0.18)
                ax.set_axisbelow(True)
                save(fig, "split_class_counts", "Class composition")

        for key, title in (("diagnostics", "Selected-data diagnostics"), ("preflight", "Preflight audit (sample scope)")):
            diagnostics = metrics.get(key) or {}
            if diagnostics:
                rows = _flatten(diagnostics)
                table("data_diagnostics" if key == "diagnostics" else "preflight_audit", title, rows, ["quantity", "value"])
                counters = [row for row in rows if isinstance(row["value"], (int, float))
                            and row["quantity"].endswith(("truncated_events", "missing_optional_events", "null_lists"))]
                if counters:
                    fig = figure(width=10, height=max(3.8, len(counters) * 0.33))
                    ax = fig.subplots()
                    ax.barh([row["quantity"] for row in counters], [row["value"] for row in counters], color=colors[1])
                    ax.invert_yaxis()
                    ax.set(xlabel="Count (scope defined in table)", title=title)
                    ax.grid(axis="x", alpha=0.18)
                    ax.set_axisbelow(True)
                    save(fig, "data_diagnostics" if key == "diagnostics" else "preflight_audit", title)

    for key in ("downstream", "performance"):
        if metrics.get(key):
            table(key, key.title(), _flatten(metrics[key]), ["quantity", "value"])

    metadata = [{"quantity": key, "value": metrics[key]} for key in
                ("view", "device", "seed", "best_epoch", "initialization") if key in metrics]
    content = ["<h1>FlowyForge · Classification</h1>", '<p class="subtitle">Preliminary results · unweighted event classification</p>', _table(metadata, ["quantity", "value"])]
    warnings = metrics.get("warnings") or []
    if isinstance(warnings, str):
        warnings = [warnings]
    if warnings:
        content.append('<aside><strong>Run notes</strong><ul>' + "".join(f"<li>{html.escape(str(warning))}</li>" for warning in warnings) + "</ul></aside>")
    content.append("<p>Confusion matrices use true classes in rows and predictions in columns. Undefined per-class scores are zero. "
                   "Balanced accuracy averages recall over classes with true support; macro F1 averages every configured class. "
                   "These unweighted results describe the selected sample, without cross-section or luminosity normalization. "
                   "Training curves aggregate predictions during optimization; validation and test summaries evaluate the selected checkpoint.</p>")
    if any(metrics.get(split, {}).get("binary") for split in ("validation", "test")):
        content.append("<p>Binary scores are the softmax probability of the explicitly named signal class. "
                       "Equal scores enter the ROC together; tied signal/background pairs contribute one half to AUC. "
                       "Efficiency uses a threshold with observed background acceptance no greater than the configured target, "
                       "without interpolation or splitting ties. A null threshold with status ok means reject all. "
                       "Insufficient background resolution yields an unavailable efficiency (not zero): "
                       "floor(N_background × target) must reach min_background_events. "
                       "This count rule is a resolution safeguard, not a confidence interval; zero observed background "
                       "does not establish zero population background acceptance. Thresholds describe each evaluated split's "
                       "empirical ROC and are not deployment thresholds calibrated on validation. "
                       "Checkpoint selection uses validation loss only; test must not guide hyperparameter choices.</p>")
    for title, rows, columns in tables:
        if title.startswith("Evaluation summary"):
            content.append(f"<section><h2>{html.escape(title)}</h2>{_table(rows, columns)}</section>")
    for title, path in figures:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(f'<section><h2>{html.escape(title)}</h2><img alt="{html.escape(title, quote=True)}" src="data:image/png;base64,{encoded}"></section>')
    for title, rows, columns in tables:
        if title.startswith("Evaluation summary"):
            continue
        if title.startswith(("Learning history", "Selected-data diagnostics", "Preflight audit")):
            content.append(f"<details><summary>{html.escape(title)}</summary>{_table(rows, columns)}</details>")
        else:
            content.append(f"<section><h2>{html.escape(title)}</h2>{_table(rows, columns)}</section>")
    if metrics.get("config"):
        config = html.escape(json.dumps(metrics["config"], indent=2, default=str, ensure_ascii=False))
        content.append(f"<details><summary>Resolved configuration</summary><pre>{config}</pre></details>")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FlowyForge classification report</title><style>
body{font:15px/1.6 system-ui,sans-serif;color:#20344b;background:#eef3f8;margin:0;padding:32px}
main{max-width:1180px;margin:auto;background:white;padding:36px;border-radius:12px}
h1{font-size:30px;margin:0;color:#153d67}h2{font-size:21px;color:#153d67;margin-top:28px}
.subtitle{color:#61758c;margin-top:4px}section{border-top:1px solid #e2e9f0;margin-top:28px}
.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:9px 12px;border-bottom:1px solid #e2e9f0;text-align:left}th{background:#edf3f9}
tr:nth-child(even){background:#f8fafc}img{display:block;width:100%;height:auto;margin:auto}
aside{background:#fff4df;border-left:4px solid #df8b29;padding:14px 20px;margin:24px 0}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:18px}
details{margin-top:30px}summary{cursor:pointer;font-weight:bold}
@media(max-width:600px){body{padding:8px}main{padding:16px}th,td{padding:6px}}
@media print{body{padding:0;background:white}main{padding:0}section,img{break-inside:avoid}}
</style></head><body><main>""" + "\n".join(content) + "</main></body></html>\n"
    report = output / "report.html"
    report.write_text(page, encoding="utf-8")
    return report


__all__ = ["compute_confusion_metrics", "compute_binary_metrics", "write_classification_report",
           "EvaluationScoreWriter", "read_evaluation_scores", "compare_classification_runs"]
