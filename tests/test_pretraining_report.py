"""Synthetic validation-only reconstruction diagnostics and legacy checkpoints."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from flowyforge.analysis import pretraining_report as reports
from flowyforge.data_plugins.collide_v2.provenance import sha256_file
from flowyforge.data_plugins.collide_v2.streaming import StreamingDataset
from flowyforge.training import pretraining_run as runner
from test_masked_pretraining import make_run_config, event, row
from flowyforge.data_plugins.collide_v2.object_dataset import collate_event_batches


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("pretraining_report")
    path, partitions = make_run_config(root)
    config = runner.load_pretraining_config(path, {"epochs": 2})
    config["reporting"].update(max_events=8, bins=4, min_bin_count=2)
    # Deliberately choose an epoch earlier than the last, to exercise decoder selection.
    original = runner._epoch
    def controlled(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["epoch"] == 2 and kwargs.get("optimizer") is None:
            result["loss"] += 100
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "_epoch", controlled)
        run_result = runner.run_pretraining(config)
    assert run_result["best_epoch"] == 1
    return Path(config["output_dir"]), partitions


def test_automatic_best_epoch_report_and_validation_only(trained, monkeypatch, tmp_path):
    run, partitions = trained
    report = json.loads((run / "pretraining_report/report.json").read_text())
    assert report["inference"]["status"] == "ok"
    assert report["inference"]["state_source"] == "best_model_state_dict"
    assert report["inference"]["epoch"] == 1
    checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
    assert checkpoint["epoch"] == 2 and checkpoint["best_model_epoch"] == 1
    assert any(not torch.equal(v, checkpoint["model_state_dict"][k]) for k, v in checkpoint["best_model_state_dict"].items() if k.startswith("decoders."))
    assert report["inference"]["sample_loss"]["loss"] == pytest.approx(report["validation"]["loss"], rel=1.e-6)
    target_ids = {e["id"] for p in partitions["processes"] for e in p["events"] if e["split"] == "validation"}
    rows = pq.read_table(run / "pretraining_report/predictions.parquet").to_pylist()
    assert rows and {r["event_id"] for r in rows} <= target_ids
    assert {r["split"] for r in rows} == {"validation"}
    assert {r["epoch"] for r in rows} == {1}
    assert all(r["object_index"] >= 0 for r in rows)
    assert all(r["true_GeV"] == pytest.approx(np.expm1(r["true_transformed"])) for r in rows)
    assert len(rows) == sum(f["masked_count"] for f in report["validation"]["features"].values())
    for name in report["figures"]:
        assert (run / f"pretraining_report/{name}.png").stat().st_size > 0
        assert (run / f"pretraining_report/{name}.pdf").stat().st_size > 0
    assert (run / "pretraining_report/pretraining_report.html").exists()
    assert report["embeddings"]["count"] == 8
    before = {p.name: sha256_file(p) for p in run.iterdir() if p.is_file()}
    original = StreamingDataset.__init__
    def guarded(self, manifest, split, **kwargs):
        assert split == "validation"
        original(self, manifest, split, **kwargs)
    monkeypatch.setattr(StreamingDataset, "__init__", guarded)
    # Batch size changes cannot change sample IDs or validation masks.
    state = torch.get_rng_state().clone()
    repeated = reports.generate_pretraining_report(run, output_dir=tmp_path / "repeat", options={"batch_size": 1})
    assert torch.equal(torch.get_rng_state(), state)
    assert repeated["inference"]["sample_event_ids_sha256"] == report["inference"]["sample_event_ids_sha256"]
    rows2 = pq.read_table(tmp_path / "repeat/predictions.parquet").to_pylist()
    key = lambda r: (r["event_id"], r["collection"], r["feature"], r["object_index"])
    first, second = sorted(rows, key=key), sorted(rows2, key=key)
    assert [key(r) for r in first] == [key(r) for r in second]
    np.testing.assert_allclose([r["predicted_transformed"] for r in first], [r["predicted_transformed"] for r in second], atol=1.e-6)
    assert before == {p.name: sha256_file(p) for p in run.iterdir() if p.is_file()}
    with pytest.raises(FileExistsError):
        reports.generate_pretraining_report(run)
    with pytest.raises(FileExistsError):
        reports.generate_pretraining_report(run, output_dir=run, overwrite=True)


def test_legacy_selection_and_mismatched_encoder_rejected(trained):
    checkpoint = torch.load(trained[0] / "checkpoint.pt", weights_only=True)
    legacy = copy.deepcopy(checkpoint)
    legacy.pop("best_model_state_dict")
    legacy.pop("best_model_epoch")
    assert reports.selected_model_state(legacy)[0] is None
    legacy["epoch"] = legacy["best_epoch"]
    legacy["model_state_dict"] = checkpoint["best_model_state_dict"]
    assert reports.selected_model_state(legacy)[1] == "legacy_last_equals_best"
    bad = copy.deepcopy(checkpoint)
    bad["best_model_epoch"] += 1
    with pytest.raises(ValueError, match="epoch"):
        reports.selected_model_state(bad)
    bad = copy.deepcopy(checkpoint)
    k = next(k for k in bad["best_model_state_dict"] if k.startswith("encoder."))
    bad["best_model_state_dict"][k] = bad["best_model_state_dict"][k] + 1
    with pytest.raises(ValueError, match="encoder differs"):
        reports.selected_model_state(bad)


def test_legacy_history_only_cli_and_incompatible_artifacts(trained, tmp_path):
    import shutil
    run = tmp_path / "legacy"
    run.mkdir()
    for name in ("metrics.json", "history.json", "resolved_config.yaml", "checkpoint.pt"):
        shutil.copy2(trained[0] / name, run / name)
    checkpoint = torch.load(run / "checkpoint.pt", weights_only=True)
    checkpoint.pop("best_model_state_dict")
    checkpoint.pop("best_model_epoch")
    torch.save(checkpoint, run / "checkpoint.pt")
    report = reports.generate_pretraining_report(run)
    assert report["inference"]["status"] == "unavailable"
    assert "no decoder" in report["inference"]["reason"]
    assert not (run / "pretraining_report/predictions.parquet").exists()
    command = [sys.executable, "scripts/report_pretraining.py", "--run-dir", str(run), "--history-only", "--overwrite"]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "test excluded" in done.stdout
    checkpoint["best_epoch"] = 2
    torch.save(checkpoint, run / "checkpoint.pt")
    with pytest.raises(ValueError, match="different runs or epochs"):
        reports.generate_pretraining_report(run, overwrite=True)
    # A failed regeneration leaves the existing report untouched.
    assert json.loads((run / "pretraining_report/report.json").read_text())["inference"]["status"] == "disabled"


def test_invalid_padding_zero_and_empty_diagnostics(trained, tmp_path, monkeypatch):
    checkpoint = torch.load(trained[0] / "checkpoint.pt", weights_only=True)
    checkpoint["config"]["pretraining"]["probability"] = 1.
    raw = row(count=3)
    raw["FullReco_PUPPIPart_PT"] = [float("nan"), 0., 5.]
    del raw["FullReco_JetPuppiAK4_Mass"]
    events = [event(0, raw=raw), event(1, count=0)]
    partitions = {"processes": [{"events": [{"id": e.metadata["event_id"], "split": "validation"} for e in events]}]}
    manifest = {"processes": [], "split_counts": {}}
    monkeypatch.setattr("flowyforge.training.classification_run._loader", lambda *a, **k: [collate_event_batches(events)])
    from flowyforge.model_plugins.foundation_encoder import MaskedReconstructionModel
    original = MaskedReconstructionModel.forward
    def guarded(self, batch):
        assert batch.targets is None
        for group in batch.objects.values():
            assert not group.feature_mask[~group.mask].any()
            assert not group.feature_mask[~group.feature_valid].any()
            assert not group.features[group.feature_mask].any()
        return original(self, batch)
    monkeypatch.setattr(MaskedReconstructionModel, "forward", guarded)
    result, ranges, moments = reports._infer(checkpoint, checkpoint["best_model_state_dict"], manifest, partitions,
                                            reports.report_options({"max_events": 2}), tmp_path)
    rows = pq.read_table(tmp_path / "predictions.parquet").to_pylist()
    assert len(rows) == 3  # PT=0/5 and jet PT; no missing mass, padding, invalid or empty-event targets.
    assert result["counts"]["empty_events"] == 1
    assert moments.n == 1
    assert reports._embedding_report(moments, tmp_path, [])["status"] == "insufficient_nonempty_finite_events"
    assert sum(r["true_GeV"] == 0 for r in rows) == 1
    figures = []
    features = reports._prediction_plots(tmp_path, ranges, reports.report_options({"bins": 4, "min_bin_count": 5}),
                                         figures, ["PUPPIPart/log1p_PT", "JetPuppiAK4/log1p_Mass"])
    assert features[0]["status"] == "insufficient_statistics"
    assert features[1]["status"] == "no_masked_targets"
    import csv
    with (tmp_path / "bias_resolution.csv").open() as stream:
        residuals = list(csv.DictReader(stream))
    assert all(r["bias"] == "" and r["resolution_std"] == "" for r in residuals)
    assert sum(int(r["count"]) for r in residuals if r["residual"] == "relative" and r["predictor"] == "predicted") == 1


def test_no_masked_targets_baseline_zero_and_embedding_moments(trained, tmp_path, monkeypatch):
    checkpoint = torch.load(trained[0] / "checkpoint.pt", weights_only=True)
    events = [event(0, count=0)]
    partitions = {"processes": [{"events": [{"id": events[0].metadata["event_id"], "split": "validation"}]}]}
    monkeypatch.setattr("flowyforge.training.classification_run._loader", lambda *a, **k: [collate_event_batches(events)])
    result, ranges, moments = reports._infer(checkpoint, checkpoint["best_model_state_dict"], {"processes": [], "split_counts": {}},
                                            partitions, reports.report_options(), tmp_path)
    assert result["counts"]["batches_without_masked_targets"] == 1
    assert result["sample_loss"]["loss"] is None and ranges == {}
    assert pq.read_table(tmp_path / "predictions.parquet").num_rows == 0
    data = np.array([[1., 2.], [2., 4.], [3., 6.]])
    moments = reports._Moments(2)
    for values in (data[:1], data[1:]):
        moments.add(values)
    np.testing.assert_allclose(moments.m2 / (moments.n - 1), np.cov(data, rowvar=False))
    summary = reports._embedding_report(moments, tmp_path, [])
    assert summary["effective_rank"] == pytest.approx(1.)
    assert np.isnan(reports._physical([1000.])[0])
    assert reports._physical([-1.])[0] < 0
    history = [{"epoch": 1, "train": {"features": {}, "loss": None}, "validation": {
        "features": {"jet/mass": {"huber": 0., "masked_count": 2}}, "loss": 0.,
        "training_constant": {"loss": 0., "features": {"jet/mass": {"huber": 0.}}}}}]
    reports._history_report(history, tmp_path, [])  # undefined 0/0 improvement is a blank cell, never inf
    assert "inf" not in (tmp_path / "feature_history.csv").read_text()


def test_sampling_and_options(trained):
    partitions = copy.deepcopy(trained[1])
    first = reports._sample_ids(partitions, 3, 5)
    partitions["processes"].reverse()
    for process in partitions["processes"]:
        process["events"].reverse()
    assert reports._sample_ids(partitions, 3, 5) == first
    for bad in ({"max_events": 0}, {"inference": "yes"}, {"seed": -1}, {"bins": 1}, {"min_bin_count": 1}, {"unknown": True}):
        with pytest.raises(ValueError):
            reports.report_options(bad)
