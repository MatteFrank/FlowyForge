"""Small report tests using aggregate counts only."""

import csv
from pathlib import Path

import pytest

from flowyforge.analysis.classification_report import (
    compute_confusion_metrics, write_classification_report,
)


def test_confusion_metrics_handle_missing_classes():
    result = compute_confusion_metrics([[3, 1, 0], [2, 4, 0], [0, 0, 0]],
                                       {"signal": 0, "background": 1, "absent": 2})
    assert result["num_events"] == 10
    assert result["accuracy"] == pytest.approx(0.7)
    assert result["balanced_accuracy"] == pytest.approx((3 / 4 + 4 / 6) / 2)
    assert result["per_class"][0]["precision"] == pytest.approx(3 / 5)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 8 / 11) / 3)
    assert result["per_class"][2]["f1"] == 0
    assert compute_confusion_metrics([[0]], {"empty": 0})["accuracy"] == 0
    with pytest.raises(ValueError, match="integer"):
        compute_confusion_metrics([[0.5]], {"invalid": 0})
    with pytest.raises(ValueError, match="contiguous"):
        compute_confusion_metrics([[1]], {"invalid": 1})


def test_report_writes_shareable_artifacts_from_aggregates(tmp_path: Path):
    name = '<signal>&"'
    metrics = {
        "label_map": {name: 0, "background": 1},
        "device": "cpu", "seed": 17, "best_epoch": 1,
        "history": [{"epoch": 1, "train_loss": 0.7, "train_accuracy": 0.6,
                     "validation": {"loss": 0.5, "accuracy": 0.75}, "duration_seconds": 0.2}],
        "test": {"confusion_matrix": [[3, 1], [0, 4]], "loss": 0.4},
        "split_class_counts": {"train": {name: 16, "background": 16},
                               "validation": {name: 4, "background": 4},
                               "test": {name: 4, "background": 4}},
        "diagnostics": {"scope": "selected data", "Jets": {"truncated_events": 2, "missing_values": 0}},
        "warnings": ["<check units>"], "config": {"mode": "synthetic"},
    }
    report = write_classification_report(metrics, tmp_path)
    document = report.read_text(encoding="utf-8")
    assert report == tmp_path / "report.html"
    assert "data:image/png;base64," in document
    assert "&lt;signal&gt;&amp;&quot;" in document
    assert "<signal>" not in document
    assert "&lt;check units&gt;" in document
    with (tmp_path / "tables/summary.csv").open(newline="", encoding="utf-8") as stream:
        summary = list(csv.DictReader(stream))
    assert summary[0]["num_events"] == "8"
    assert float(summary[0]["accuracy"]) == pytest.approx(0.875)
    for stem in ("learning_curves", "confusion_test_counts", "confusion_test_normalized",
                 "per_class_test", "split_class_counts", "data_diagnostics"):
        assert (tmp_path / "plots" / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / "plots" / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
    assert (tmp_path / "tables/per_class_test.csv").exists()
    assert (tmp_path / "tables/confusion_test.csv").exists()


def test_report_accepts_absent_optional_fields(tmp_path: Path):
    report = write_classification_report({}, tmp_path)
    assert report.exists()
    assert "Preliminary results" in report.read_text(encoding="utf-8")
