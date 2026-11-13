"""Epoch-boundary CPU replay, interrupted commits, and inference compatibility."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from flowyforge.training import classification_run as runner
from flowyforge.training.checkpointing import state_digest
from flowyforge.analysis.classification_report import compare_classification_runs, compare_downstream_runs, read_evaluation_scores
from flowyforge.data_plugins.collide_v2.provenance import sha256_file
from test_downstream_transfer import pilot, config_for  # reusable synthetic two-view data/encoder fixture


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def configured(pilot, mode, output, epochs=3, view="FullReco"):
    config = config_for(pilot, mode if mode != "legacy" else "scratch", output=output, view=view)
    if mode == "legacy":
        config.pop("downstream")
    config["training"]["epochs"] = epochs
    return config


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if "second" not in k}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


@pytest.fixture(scope="module")
def resumed_runs(pilot, tmp_path_factory):
    root = tmp_path_factory.mktemp("resumed_classification")
    results = {}
    for mode in ("legacy", "scratch", "linear_probe", "finetune"):
        full_config = configured(pilot, mode, root / f"{mode}_continuous")
        continuous = runner.run_classification(full_config)
        config = configured(pilot, mode, root / mode, epochs=1)
        original = runner.run_classification(config)
        before = load(root / mode / "last_checkpoint.pt")
        config["training"]["epochs"] = 3
        first_forward = []
        original_epoch = runner._run_epoch
        def audited(model, loader, device, label_map, **kwargs):
            if kwargs.get("optimizer") is not None and not first_forward:
                first_forward.append(True)
                assert state_digest(model.state_dict()) == state_digest(before["model_state_dict"])
                assert state_digest(kwargs["optimizer"].state_dict()) == state_digest(before["optimizer_state_dict"])
                assert kwargs["description"] == "epoch 2 train"
                if mode == "linear_probe":
                    model.train()
                    assert not any(m.training for m in model.encoder.modules())
                    assert not any(p.requires_grad for p in model.encoder.parameters())
                    assert {id(p) for g in kwargs["optimizer"].param_groups for p in g["params"]} == {id(p) for p in model.head.parameters()}
            if loader.dataset.split == "test":
                assert load(root / mode / "last_checkpoint.pt")["epoch"] == 3
                assert state_digest(model.state_dict()) == state_digest(load(root / mode / "checkpoint.pt")["model_state_dict"])
            return original_epoch(model, loader, device, label_map, **kwargs)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runner, "_run_epoch", audited)
            # Any accidental pretrained reinitialization during resume would fail.
            patch.setattr("flowyforge.model_plugins.foundation_encoder.load_encoder", lambda *a, **k: pytest.fail("Resume reimported encoder.pt"))
            resumed = runner.run_classification(config, resume=root / mode / "last_checkpoint.pt")
        assert first_forward
        results[mode] = (continuous, original, resumed, before)
    return root, results


@pytest.mark.parametrize("mode", ["legacy", "scratch", "linear_probe", "finetune"])
def test_exact_replay_all_modes_and_frozen_or_updated_encoder(resumed_runs, mode):
    root, results = resumed_runs
    continuous, original, resumed, before = results[mode]
    last = load(root / mode / "last_checkpoint.pt")
    full = load(root / f"{mode}_continuous/last_checkpoint.pt")
    assert last["epoch"] == full["epoch"] == 3
    for key in ("model_state_dict", "optimizer_state_dict", "rng"):
        assert state_digest(last[key]) == state_digest(full[key]), key
    assert without_timing(resumed["history"]) == without_timing(continuous["history"])
    assert without_timing(resumed["test"]) == without_timing(continuous["test"])
    assert resumed["best_epoch"] == continuous["best_epoch"]
    assert resumed["best_validation_loss"] == continuous["best_validation_loss"]
    assert resumed["initial_state_sha256"] == original["initial_state_sha256"]
    assert resumed["diagnostics"] == continuous["diagnostics"]
    metadata = resumed["training_run"]
    assert metadata["resumed"] and metadata["start_epoch"] == 2 and metadata["end_epoch"] == 3
    assert metadata["epochs_completed"] == 3 and metadata["epochs_this_invocation"] == 2
    assert len(metadata["sessions"]) == 2
    assert metadata["resume_source"]["saved_epoch"] == 1
    assert json.loads((root / mode / "history.json").read_text()) == resumed["history"]
    best = load(root / mode / "checkpoint.pt")
    assert best["format_version"] == (1 if mode == "legacy" else 2)
    assert best["config"]["training"]["epochs"] == 3
    assert best["training_protocol"] == resumed["training_protocol"]
    assert best["training_run"] == metadata
    assert state_digest(best["model_state_dict"]) == state_digest(last["best_checkpoint"]["model_state_dict"])
    archive = list((root / mode / "resume_archive").glob("run-*"))
    assert len(archive) == 1 and (archive[0] / "scores/test/manifest.json").exists()
    assert json.loads((archive[0] / "metrics.json").read_text())["history"] == original["history"]
    for split in ("validation", "test"):
        a = read_evaluation_scores(root / mode / resumed[split]["score_manifest"])
        b = read_evaluation_scores(root / f"{mode}_continuous" / continuous[split]["score_manifest"])
        assert dict(zip(a[1], a[3])) == dict(zip(b[1], b[3]))
    if mode in {"linear_probe", "finetune"}:
        changed = any(not torch.equal(v, before["model_state_dict"][k]) for k, v in last["model_state_dict"].items() if k.startswith("encoder."))
        assert changed == (mode == "finetune")
    assert "Training_Run" in (root / mode / "report.html").read_text()  # report metadata table


def test_resumed_comparators_evaluate_report_and_cli(pilot, resumed_runs, tmp_path):
    root, _ = resumed_runs
    compare_downstream_runs([root / "scratch_continuous", root / "linear_probe", root / "finetune"], tmp_path / "transfer")
    config = configured(pilot, "scratch", tmp_path / "l1t", epochs=1, view="L1T")
    runner.run_classification(config)
    # Use the actual CLI and the checkpoint.pt alias; it must use its resume sibling.
    command = [sys.executable, "scripts/run_classification.py", "--config", str(pilot[1]), "--stage", "train",
               "--view", "L1T", "--output-dir", str(tmp_path / "l1t"), "--epochs", "3",
               "--resume", str(tmp_path / "l1t/checkpoint.pt")]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    compare_classification_runs([root / "scratch", tmp_path / "l1t"], tmp_path / "views")
    for mode in ("legacy", "finetune"):
        result = runner.evaluate_run(root / mode / "checkpoint.pt", output_dir=tmp_path / f"eval_{mode}", device="cpu")
        assert result["training_run"]["resumed"]
        done = subprocess.run([sys.executable, "scripts/run_classification.py", "--stage", "report", "--output-dir", str(tmp_path / f"eval_{mode}")], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    compare_downstream_runs([root / "scratch", root / "linear_probe", tmp_path / "eval_finetune"], tmp_path / "reeval_transfer")


def test_interruption_mid_epoch_uses_only_completed_state(pilot, tmp_path, monkeypatch):
    config = configured(pilot, "finetune", tmp_path / "interrupted")
    original = runner._run_epoch
    seen = []
    def interrupted(model, loader, device, label_map, **kwargs):
        seen.append(loader.dataset.split)
        if kwargs["description"] == "epoch 2 train":
            # A partial epoch can mutate both weights and optimizer before interruption.
            original(model, [next(iter(loader))], device, label_map, **kwargs)
            raise RuntimeError("simulated interruption")
        return original(model, loader, device, label_map, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_run_epoch", interrupted)
        with pytest.raises(RuntimeError, match="simulated"):
            runner.run_classification(config)
    assert "test" not in seen
    assert load(tmp_path / "interrupted/last_checkpoint.pt")["epoch"] == 1
    assert len(json.loads((tmp_path / "interrupted/history.json").read_text())) == 1
    resumed = runner.run_classification(config, resume=tmp_path / "interrupted/last_checkpoint.pt")
    config["output_dir"] = str(tmp_path / "continuous")
    full = runner.run_classification(config)
    assert without_timing(resumed["history"]) == without_timing(full["history"])
    assert state_digest(load(tmp_path / "interrupted/last_checkpoint.pt")["model_state_dict"]) == state_digest(load(tmp_path / "continuous/last_checkpoint.pt")["model_state_dict"])


def test_atomic_commit_recovers_missing_best_and_keeps_earlier_best(pilot, tmp_path, monkeypatch):
    config = configured(pilot, "scratch", tmp_path / "commit", epochs=2)
    original = runner._save_checkpoint
    def fail_best(path, payload):
        if path.name == "checkpoint.pt":
            raise OSError("simulated crash after last commit")
        original(path, payload)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_save_checkpoint", fail_best)
        with pytest.raises(OSError, match="simulated"):
            runner.run_classification(config)
    assert load(tmp_path / "commit/last_checkpoint.pt")["epoch"] == 1
    assert not (tmp_path / "commit/checkpoint.pt").exists()
    original_epoch = runner._run_epoch
    def worse(model, loader, device, labels, **kwargs):
        result = original_epoch(model, loader, device, labels, **kwargs)
        if kwargs["description"] == "epoch 2 validation":
            result["loss"] += 100  # Force the original best to survive the extension.
        return result
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_run_epoch", worse)
        result = runner.run_classification(config, resume=tmp_path / "commit/last_checkpoint.pt")
    assert result["best_epoch"] == 1
    last = load(tmp_path / "commit/last_checkpoint.pt")
    assert last["epoch"] == 2
    assert state_digest(last["model_state_dict"]) != state_digest(last["best_checkpoint"]["model_state_dict"])
    assert result["validation"]["loss"] == pytest.approx(result["history"][0]["validation"]["loss"])


@pytest.mark.parametrize("change", ["mode", "view", "batch", "lr", "limits", "architecture", "workers", "shorter", "partition", "data"])
def test_incompatible_config_rejected_without_overwrites(pilot, resumed_runs, tmp_path, change):
    root, _ = resumed_runs
    directory = tmp_path / "copy"
    shutil.copytree(root / "scratch", directory)
    config = configured(pilot, "scratch", directory, epochs=4)
    if change == "mode":
        config.pop("downstream")
    elif change == "view":
        config["data"]["view"] = "L1T"
    elif change == "batch":
        config["training"]["batch_size"] += 1
    elif change == "lr":
        config["training"]["learning_rate"] *= .1
    elif change == "limits":
        config["data"]["collection_limits"]["PUPPIPart"] += 1
    elif change == "architecture":
        config["model"]["num_latents"] += 1
    elif change == "workers":
        config["training"]["num_workers"] = 1
    elif change == "shorter":
        config["training"]["epochs"] = 2
    elif change == "partition":
        config["partition_manifest"] = "changed.json"
    elif change == "data":
        config["source"]["processes"] = {"a": "different.parquet", "b": "changed.parquet"}
    before = {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="resume configuration|TOTAL horizon"):
        runner.run_classification(config, resume=directory / "last_checkpoint.pt")
    assert before == {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}


@pytest.mark.parametrize("damage", ["optimizer", "rng", "digest", "truncated", "old", "manifest", "partition"])
def test_corrupt_or_old_resume_rejected(pilot, resumed_runs, tmp_path, damage):
    root, _ = resumed_runs
    directory = tmp_path / "copy"
    shutil.copytree(root / "finetune", directory)
    config = configured(pilot, "finetune", directory, epochs=4)
    path = directory / "last_checkpoint.pt"
    saved = load(path)
    if damage in {"optimizer", "rng"}:
        saved["optimizer_state_dict" if damage == "optimizer" else "rng"] = {}
        saved["state_sha256"] = state_digest({k: v for k, v in saved.items() if k != "state_sha256"})
        torch.save(saved, path)
    elif damage == "digest":
        saved["model_state_dict"][next(iter(saved["model_state_dict"]))].add_(1)
        torch.save(saved, path)
    elif damage == "truncated":
        path.write_bytes(b"corrupt archive")
    elif damage == "old":
        path.unlink()
        path = directory / "checkpoint.pt"
    elif damage in {"manifest", "partition"}:
        p = directory / ("dataset_manifest.json" if damage == "manifest" else "partitions.json")
        value = json.loads(p.read_text()); value["seed"] += 1; p.write_text(json.dumps(value))
    before = {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}
    with pytest.raises(ValueError):
        runner.run_classification(config, resume=path)
    assert before == {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}


def test_legacy_manifest_and_final_evaluation_retry(pilot, tmp_path, monkeypatch):
    config = configured(pilot, "legacy", tmp_path / "legacy", epochs=1)
    config["partition_manifest"] = None
    config["evaluation"]["signal_process"] = None
    def interrupted(*args, **kwargs):
        raise RuntimeError("final evaluation interrupted")
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_evaluate_split", interrupted)
        with pytest.raises(RuntimeError, match="final evaluation interrupted"):
            runner.run_classification(config)
    before = load(tmp_path / "legacy/last_checkpoint.pt")
    # No new training for an equal total horizon; only final evaluation is retried.
    original_epoch = runner._run_epoch
    def no_training(*args, **kwargs):
        assert kwargs.get("optimizer") is None
        return original_epoch(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runner, "_run_epoch", no_training)
        result = runner.run_classification(config, resume=tmp_path / "legacy/last_checkpoint.pt")
    assert result["training_run"]["epochs_this_invocation"] == 0
    assert state_digest(before["model_state_dict"]) == state_digest(load(tmp_path / "legacy/last_checkpoint.pt")["model_state_dict"])
    config["training"]["epochs"] = 2
    runner.run_classification(config, resume=tmp_path / "legacy/last_checkpoint.pt")


def test_changed_source_at_same_path_rejected(pilot, resumed_runs, tmp_path):
    import os
    import pyarrow.parquet as pq
    root, _ = resumed_runs
    directory = tmp_path / "copy"
    shutil.copytree(root / "scratch", directory)
    config = configured(pilot, "scratch", directory, epochs=4)
    manifest = json.loads((directory / "dataset_manifest.json").read_text())
    path = Path(manifest["processes"][0]["files"][0]["path"])
    original, stat = path.read_bytes(), path.stat()
    before = sha256_file(directory / "last_checkpoint.pt")
    try:
        table = pq.read_table(path)
        pq.write_table(table.slice(1), path)  # original path/config, different source rows
        with pytest.raises(ValueError):
            runner.run_classification(config, resume=directory / "last_checkpoint.pt")
        assert sha256_file(directory / "last_checkpoint.pt") == before
    finally:
        path.write_bytes(original)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_worker_restart_smoke(pilot, tmp_path):
    config = configured(pilot, "scratch", tmp_path / "worker", epochs=1)
    config["training"]["num_workers"] = 1
    runner.run_classification(config)
    config["training"]["epochs"] = 2
    result = runner.run_classification(config, resume=tmp_path / "worker/last_checkpoint.pt")
    assert [h["epoch"] for h in result["history"]] == [1, 2]
    assert all(h["train_num_events"] == 24 for h in result["history"])
    assert result["split_sizes"] == {"train": 24, "validation": 8, "test": 8}
