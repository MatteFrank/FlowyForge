"""Small report tests using aggregate counts only."""

import csv
from pathlib import Path

import pytest

from flowyforge.analysis.classification_report import (
    EvaluationScoreWriter, compute_binary_metrics, compute_confusion_metrics,
    read_evaluation_scores, write_classification_report,
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


def test_binary_roc_auc_groups_ties_and_never_splits_them_at_operating_point():
    result = compute_binary_metrics([0.9, 0.8, 0.8, 0.1], [1, 0, 1, 0], signal_label=1,
                                     background_acceptance=0.5, min_background_events=1)
    assert result["roc_auc"] == pytest.approx(0.875)
    assert result["num_signal"] == result["num_background"] == 2
    assert [(p["false_positive_rate"], p["true_positive_rate"]) for p in result["roc"]] == [(0, 0), (0, .5), (.5, 1), (1, 1)]
    assert result["operating_point"]["threshold"] == .8
    assert result["operating_point"]["signal_efficiency"] == 1
    tied = compute_binary_metrics([.5] * 8, [0, 1] * 4, signal_label=0,
                                   background_acceptance=.5, min_background_events=1)
    assert tied["roc_auc"] == .5
    assert tied["operating_point"]["status"] == "ok"
    assert tied["operating_point"]["signal_efficiency"] == 0
    assert tied["operating_point"]["threshold"] is None
    assert tied["operating_point"]["accepted_background"] == 0
    assert tied["operating_point"]["tie_limited"]
    reordered = compute_binary_metrics([.8, .1, .9, .8], [1, 0, 1, 0], signal_label=1,
                                        background_acceptance=.5, min_background_events=1)
    assert result == reordered


def test_binary_roc_known_rankings_missing_classes_and_low_background_resolution():
    for scores, auc in (([.9, .8, .2, .1], 1.), ([.1, .2, .8, .9], 0.)):
        result = compute_binary_metrics(scores, [1, 1, 0, 0], signal_label=1)
        assert result["roc_auc"] == auc
        assert result["operating_point"]["signal_efficiency"] is None
        assert result["operating_point"]["status"] == "insufficient_background_resolution"
    for labels in ([], [0, 0], [1, 1]):
        result = compute_binary_metrics([.2] * len(labels), labels, signal_label=1)
        assert result["roc_auc"] is None
        assert result["status"] == "missing_class" and result["roc"] == []
    # A fractional floating-point budget such as .29 * 100 must admit 29 events.
    scores = [1 - i / 101 for i in range(101)]
    result = compute_binary_metrics(scores, [0] * 100 + [1], signal_label=1,
                                     background_acceptance=.29, min_background_events=1)
    assert result["operating_point"]["accepted_background"] == 29
    with pytest.raises(ValueError, match="finite"):
        compute_binary_metrics([float('nan')], [0], signal_label=1)
    with pytest.raises(ValueError, match="background_acceptance"):
        compute_binary_metrics([.5], [0], signal_label=1, background_acceptance=0)


def test_score_shards_are_bounded_complete_and_protected(tmp_path):
    context = {"signal_process": "signal", "signal_label": 1}
    writer = EvaluationScoreWriter(tmp_path, "test", context=context, shard_rows=3)
    for start in (0, 5):
        ids = [{"event_id": f"origin-v1:{index:064x}"} for index in range(start, start + 5)]
        writer.append(ids, [0, 1, 0, 1, 0], [.2, .6, .1, .8, .3])
        assert len(writer.rows) < 3
    relative = writer.finish()
    manifest, ids, labels, scores = read_evaluation_scores(tmp_path / relative)
    assert manifest["num_rows"] == len(ids) == len(labels) == len(scores) == 10
    assert [shard["num_rows"] for shard in manifest["shards"]] == [3, 3, 3, 1]
    with pytest.raises(FileExistsError):
        EvaluationScoreWriter(tmp_path, "test", context=context)
    file = tmp_path / "scores/test" / manifest["shards"][0]["path"]
    file.write_bytes(file.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="content changed"):
        read_evaluation_scores(tmp_path / relative)


def test_binary_report_retains_classification_and_exports_roc(tmp_path):
    binary = compute_binary_metrics([.9, .8, .2, .1], [1, 0, 1, 0], signal_label=1,
                                     background_acceptance=.01, min_background_events=10)
    binary.update(signal_process="ttbar", background_process="Wjets")
    report = write_classification_report({"view": "FullReco", "label_map": {"Wjets": 0, "ttbar": 1},
        "test": {"confusion_matrix": [[1, 1], [1, 1]], "loss": .5, "binary": binary}}, tmp_path)
    document = report.read_text()
    assert "ttbar" in document and "insufficient_background_resolution" in document
    assert "FullReco" in document
    assert "not a confidence interval" in document
    for name in ("roc_test", "confusion_test_counts"):
        assert (tmp_path / "plots" / (name + ".png")).exists()
    with (tmp_path / "tables/binary_test.csv").open() as stream:
        row = next(csv.DictReader(stream))
        assert row["signal_efficiency"] == ""
        assert row["num_signal"] == row["num_background"] == "2"
    assert (tmp_path / "tables/roc_test.csv").exists()
