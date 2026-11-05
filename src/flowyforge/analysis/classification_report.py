"""Shareable classification reports from streaming aggregate statistics."""

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path
from typing import Any

import numpy as np


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

    metadata = [{"quantity": key, "value": metrics[key]} for key in ("device", "seed", "best_epoch") if key in metrics]
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


__all__ = ["compute_confusion_metrics", "write_classification_report"]
