"""One tiny CPU integration run; no external datasets or downloads."""

import json
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from flowyforge.core.typing import EventBatch, ObjectGroup
from flowyforge.training.classification_run import (
    batch_to_device, evaluate_run, load_run_config, resolve_device, run_classification,
)
from flowyforge.analysis.classification_report import compare_classification_runs, read_evaluation_scores
from flowyforge.data_plugins.collide_v2.streaming import create_partition_manifest, save_partition_manifest


REPO = Path(__file__).resolve().parents[1]


def test_batch_device_preserves_masks_targets_and_provenance():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    batch = EventBatch({"jet": ObjectGroup("jet", torch.ones(2, 3, 4),
                                          torch.tensor([[True, False, False], [True, True, False]]))},
                       torch.tensor([0, 1]), {"source": "kept on CPU"})
    for device in devices:
        moved = batch_to_device(batch, device)
        assert moved.objects["jet"].features.device.type == device.type
        assert moved.objects["jet"].mask.device.type == device.type
        assert moved.targets.device.type == device.type
        assert moved.objects["jet"].mask.dtype == torch.bool
        assert moved.metadata == batch.metadata
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="unavailable"):
            resolve_device("cuda")


def test_invalid_run_config_fails_before_reading_data():
    path = REPO / "configs/runs/classification.yaml"
    with pytest.raises(ValueError, match="small mode"):
        load_run_config(path)
    with pytest.raises(ValueError, match="max_events_per_class"):
        load_run_config(path, {"mode": "eos", "max_events_per_class": 2})
    with pytest.raises(ValueError, match="batch_size"):
        load_run_config(path, {"mode": "eos", "batch_size": 0})
    with pytest.raises(ValueError, match="partition_manifest"):
        load_run_config(path, {"mode": "eos", "partition_manifest": False})
    config = load_run_config(path, {"mode": "eos", "max_events_per_class": 0})
    assert config["source"]["max_events_per_class"] is None
    assert config["source"]["root"] == "/eos/project/f/foundational-model-dataset/samples/"


@pytest.mark.parametrize("shared", [False, True])
def test_small_train_reload_report_and_manifest_integrity(tmp_path, shared):
    processes = {}
    for label, name in enumerate(("a", "b")):
        rows = [{"FullReco_PUPPIMET_MET": [float(i + label * 20)],
                 "FullReco_PUPPIMET_Phi": [0.1 * i]} for i in range(20)]
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=5)
        processes[name] = str(path)
    model_config = tmp_path / "model.yaml"
    model_config.write_text(yaml.safe_dump({
        "model": {"embedding_dim": 8, "num_latents": 2, "num_heads": 2, "num_layers": 1,
                  "output_dim": 8, "dropout": 0.0},
        "data": {"collection_limits": {"PUPPIPart": 0, "JetPuppiAK4": 0, "Electron": 0,
                                        "MuonTight": 0, "PhotonTight": 0, "PUPPIMET": 1}},
    }))
    output = tmp_path / "run"
    config = load_run_config(REPO / "configs/runs/classification.yaml", {
        "processes": processes, "max_events_per_class": 20, "model_config": str(model_config),
        "epochs": 1, "batch_size": 8, "device": "cpu", "output_dir": str(output),
    })
    config["training"]["cpu_threads"] = 1
    config["data"]["audit_events_per_class"] = 3
    config["data"]["shuffle_buffer"] = 4
    if shared:
        partition_path = tmp_path / "partitions.json"
        command = [sys.executable, str(REPO / "scripts/run_classification.py"),
                   "--stage", "partitions", "--model-config", str(model_config),
                   "--partition-manifest", str(partition_path), "--max-events-per-class", "20"]
        for name, path in processes.items():
            command.extend(["--process", f"{name}={path}"])
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        repeated = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        assert repeated.returncode == 1
        assert "overwrite existing partitions" in repeated.stderr
        config["partition_manifest"] = str(partition_path)
    previous_threads = torch.get_num_threads()
    try:
        preflight = run_classification(config, inspect_only=True)
        assert preflight
        assert not (output / "checkpoint.pt").exists()
        metrics = run_classification(config)
        assert metrics["best_epoch"] == 1
        assert sum(metrics["split_sizes"].values()) == 40
        assert metrics["test"]["num_events"] == metrics["split_sizes"]["test"]
        for filename in ("metrics.json", "checkpoint.pt", "history.json", "dataset_manifest.json",
                         "preflight.json", "resolved_config.yaml", "report.html"):
            assert (output / filename).is_file()
        assert list((output / "tables").glob("*.csv"))
        assert list((output / "plots").glob("*.png"))
        assert list((output / "plots").glob("*.pdf"))
        relocation = {}
        if shared:
            moved = tmp_path / "relocated"
            for name, path in processes.items():
                (moved / name).mkdir(parents=True)
                shutil.move(path, moved / name / "copied.parquet")
            relocation["data_root"] = moved
            partition_path.unlink()  # Evaluation must use the immutable run snapshot.
            assert metrics["partition_sha256"]
        loaded = evaluate_run(output / "checkpoint.pt", output_dir=tmp_path / "evaluation", device="cpu", **relocation)
        assert loaded["test"]["confusion_matrix"] == metrics["test"]["confusion_matrix"]
        assert loaded["test"]["loss"] == pytest.approx(metrics["test"]["loss"])
        with pytest.raises(FileExistsError, match="existing"):
            run_classification(config)
        if shared:
            from flowyforge.data_plugins.collide_v2.streaming import _json_digest
            saved = output / "partitions.json"
            changed = json.loads(saved.read_text())
            changed["seed"] += 1
            changed["sha256"] = _json_digest({k: v for k, v in changed.items() if k != "sha256"})
            saved.write_text(json.dumps(changed))
            with pytest.raises(ValueError, match="saved memberships"):
                evaluate_run(output / "checkpoint.pt", output_dir=tmp_path / "bad_partitions", **relocation)
        else:
            with pytest.raises(ValueError, match="Relocation requires"):
                evaluate_run(output / "checkpoint.pt", output_dir=tmp_path / "bad_relocation", data_root=tmp_path)
        manifest_path = output / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tampered"] = True
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="manifest"):
            evaluate_run(output / "checkpoint.pt", output_dir=tmp_path / "bad", device="cpu")
    finally:
        torch.set_num_threads(previous_threads)


def test_two_view_supervised_reference_cpu_scores_evaluation_and_guarded_comparison(tmp_path):
    processes = {}
    for label, name in enumerate(("Wjets", "ttbar")):
        path = tmp_path / name / "events.parquet"
        path.parent.mkdir()
        rows = [{f"{view}_PUPPIMET_{field}": [float(i + label * 20) if field == "MET" else .1]
                 for view in ("FullReco", "L1T") for field in ("MET", "Phi")} for i in range(20)]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=5)
        processes[name] = str(path)
    model_path = tmp_path / "tiny_model.yaml"
    model_path.write_text(yaml.safe_dump({
        "model": {"embedding_dim": 8, "num_latents": 2, "num_heads": 2, "num_layers": 1,
                  "output_dim": 8, "dropout": 0.0},
        "data": {"collection_limits": {"PUPPIPart": 0, "JetPuppiAK4": 0, "Electron": 0,
                                        "MuonTight": 0, "PhotonTight": 0, "PUPPIMET": 1}}}))
    partitions = create_partition_manifest({"processes": processes})
    partition_path = tmp_path / "partitions.json"
    save_partition_manifest(partitions, partition_path)
    run_path = tmp_path / "reference.yaml"
    run_path.write_text(yaml.safe_dump({"model_config": str(model_path), "partition_manifest": str(partition_path),
        "seed": 42, "sources": {"small": {"processes": processes, "max_events_per_class": None}},
        "data": {"audit_events_per_class": 2, "shuffle_buffer": 4, "read_batch_size": 3},
        "training": {"epochs": 1, "batch_size": 8, "device": "cpu", "cpu_threads": 1},
        "evaluation": {"signal_process": "ttbar", "background_acceptance": .5,
                       "min_background_events": 1, "score_shard_rows": 3}}))
    run_dirs, results = [], []
    previous_threads = torch.get_num_threads()
    try:
        for view in ("FullReco", "L1T"):
            directory = tmp_path / view
            config = load_run_config(run_path, {"view": view, "output_dir": str(directory)})
            run_classification(config, inspect_only=True)
            assert not (directory / "scores").exists()
            result = run_classification(config)
            assert result["initialization"] == "random_from_scratch"
            assert result["test"]["binary"]["signal_process"] == "ttbar"
            assert result["test"]["binary"]["signal_label"] == 1
            assert result["test"]["binary"]["num_signal"] == result["test"]["binary"]["num_background"] == 4
            checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert checkpoint["initial_state_sha256"] == result["initial_state_sha256"]
            selected = min(result["history"], key=lambda item: item["validation"]["loss"])
            assert result["best_epoch"] == selected["epoch"]
            assert all("binary" not in item["validation"] for item in result["history"])
            for split in ("validation", "test"):
                score_manifest, ids, labels, scores = read_evaluation_scores(directory / result[split]["score_manifest"])
                assert all(shard["num_rows"] <= 3 for shard in score_manifest["shards"])
                assert len(ids) == result[split]["num_events"]
                expected = {e["id"] for p in partitions["processes"] for e in p["events"] if e["split"] == split}
                assert set(ids) == expected
            run_dirs.append(directory)
            results.append(result)
        assert results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"]
        comparison = compare_classification_runs(run_dirs, tmp_path / "comparison")
        assert len(comparison["results"]) == 4
        with pytest.raises(FileExistsError, match="overwrite"):
            compare_classification_runs(run_dirs, tmp_path / "comparison")
        evaluation_dir = tmp_path / "reevaluation"
        evaluation = evaluate_run(run_dirs[0] / "checkpoint.pt", output_dir=evaluation_dir, device="cpu", batch_size=5)
        assert evaluation["test"]["binary"]["roc_auc"] == pytest.approx(results[0]["test"]["binary"]["roc_auc"])
        assert evaluation["training_protocol"] == results[0]["training_protocol"]
        # A different evaluation batch size must not masquerade as changed training.
        compare_classification_runs([evaluation_dir, run_dirs[1]], tmp_path / "reevaluation_comparison")
        with pytest.raises(FileExistsError):
            evaluate_run(run_dirs[0] / "checkpoint.pt", output_dir=evaluation_dir)
        command = [sys.executable, str(REPO / "scripts/run_classification.py"), "--stage", "compare",
                   "--run-dir", str(run_dirs[0]), "--run-dir", str(run_dirs[1]), "--output-dir", str(tmp_path / "cli_comparison")]
        cli = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        assert cli.returncode == 0, cli.stderr
        # Reject different optimization settings before producing any comparison.
        metrics_path = run_dirs[1] / "metrics.json"
        modified = copy.deepcopy(results[1])
        modified["training_protocol"]["training"]["epochs"] += 1
        metrics_path.write_text(json.dumps(modified))
        with pytest.raises(ValueError, match="training_protocol"):
            compare_classification_runs(run_dirs, tmp_path / "bad_settings")
        assert not (tmp_path / "bad_settings").exists()
        metrics_path.write_text(json.dumps(results[1]))
        # Equal class counts do not suffice: changing a scored ID is rejected.
        from flowyforge.data_plugins.collide_v2.provenance import sha256_file
        score_path = run_dirs[1] / "scores/test/manifest.json"
        score_manifest = json.loads(score_path.read_text())
        shard = score_path.parent / score_manifest["shards"][0]["path"]
        table = pq.read_table(shard)
        ids = table["event_id"].to_pylist()
        ids[0] = "origin-v1:" + "f" * 64
        table = table.set_column(0, table.schema.field(0), pa.array(ids))
        pq.write_table(table, shard)
        score_manifest["shards"][0]["sha256"] = sha256_file(shard)
        score_path.write_text(json.dumps(score_manifest))
        with pytest.raises(ValueError, match="events/labels/context"):
            compare_classification_runs(run_dirs, tmp_path / "bad_events")
        assert not (tmp_path / "bad_events").exists()
    finally:
        torch.set_num_threads(previous_threads)


def test_supervised_config_fixes_common_protocol_without_changing_features():
    path = REPO / "configs/runs/collide_supervised.yaml"
    full = load_run_config(path)
    l1t = load_run_config(path, {"view": "L1T", "output_dir": "outputs/perceiver/supervised_l1t"})
    assert full["model"] == l1t["model"]
    assert full["training"] == l1t["training"]
    assert full["source"] == l1t["source"]
    assert full["partition_manifest"] == l1t["partition_manifest"]
    assert full["output_dir"] != l1t["output_dir"]
    reference = yaml.safe_load((REPO / "configs/models/foundation_tiny.yaml").read_text())
    assert full["data"]["collection_limits"] == reference["data"]["collection_limits"]
    assert full["evaluation"]["signal_process"] == "tt0123j_5f_ckm_LO_MLM_semiLeptonic"
