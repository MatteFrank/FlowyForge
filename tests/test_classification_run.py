"""One tiny CPU integration run; no external datasets or downloads."""

import json
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
