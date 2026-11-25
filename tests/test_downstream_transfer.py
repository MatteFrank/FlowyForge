"""Small CPU transfer benchmark; real reader/model, synthetic data only."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from flowyforge.analysis.classification_report import (
    compare_classification_runs, compare_downstream_runs, read_evaluation_scores,
)
from flowyforge.data_plugins.collide_v2.provenance import sha256_file
from flowyforge.data_plugins.collide_v2.streaming import create_partition_manifest, save_partition_manifest
from flowyforge.data_plugins.collide_v2.synthetic import make_tiny_parquet_sample
from flowyforge.model_plugins.foundation_encoder import load_encoder
from flowyforge.training import classification_run as runner
from flowyforge.training.pretraining_run import load_pretraining_config, run_pretraining

REPO = Path(__file__).resolve().parents[1]
MODES = ("scratch", "linear_probe", "finetune")


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("downstream")
    make_tiny_parquet_sample(root / "data", events_per_class=20, files_per_class=1)
    source = {"processes": {name: str(root / "data" / name) for name in ("background", "signal")},
              "max_events_per_class": None}
    partitions = create_partition_manifest(source)
    save_partition_manifest(partitions, root / "partitions.json")
    model_path = root / "model.yaml"
    model_path.write_text(yaml.safe_dump({"model": {"embedding_dim": 8, "num_latents": 2, "num_heads": 2,
        "num_layers": 1, "output_dim": 8, "dropout": .4},
        "data": {"collection_limits": {"PUPPIPart": 8, "JetPuppiAK4": 3, "Electron": 2,
                                        "MuonTight": 2, "PhotonTight": 2, "PUPPIMET": 1}}}))
    raw = {"model_config": str(model_path), "partition_manifest": str(root / "partitions.json"),
           "seed": 42, "sources": {"small": source}, "data": {"read_batch_size": 4, "shuffle_buffer": 4,
           "audit_events_per_class": 2}, "training": {"epochs": 2, "batch_size": 8, "device": "cpu", "cpu_threads": 1},
           "evaluation": {"signal_process": "signal", "background_acceptance": .5, "min_background_events": 1,
                          "score_shard_rows": 3}, "downstream": {"mode": "scratch"}}
    config_path = root / "downstream.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    pretrain = copy.deepcopy(raw)
    pretrain.pop("downstream")
    pretrain["pretraining"] = {"probability": .5, "decoder_hidden_dim": 8}
    pretrain["training"]["epochs"] = 1
    pretrain_path = root / "pretraining.yaml"
    pretrain_path.write_text(yaml.safe_dump(pretrain))
    for view in ("FullReco", "L1T"):
        config = load_pretraining_config(pretrain_path, {"view": view, "output_dir": str(root / f"pretrain_{view}")})
        run_pretraining(config)
    return root, config_path, partitions


def config_for(pilot, mode, *, view="FullReco", output=None):
    root, path, _ = pilot
    overrides = {"downstream_mode": mode, "view": view, "output_dir": str(output or root / f"{view}_{mode}")}
    if mode != "scratch":
        overrides["encoder_checkpoint"] = str(root / f"pretrain_{view}/encoder.pt")
    return runner.load_run_config(path, overrides)


def test_linear_probe_is_strictly_linear_frozen_eval_and_finetune_updates_encoder(pilot):
    source = load_encoder(pilot[0] / "pretrain_FullReco/encoder.pt")
    heads = []
    for mode in MODES:
        config = config_for(pilot, mode)
        torch.manual_seed(config["seed"])
        model = runner._make_model(config, {"background": 0, "signal": 1})
        assert type(model.head) is torch.nn.Linear
        heads.append(runner._initial_state_digest(model.head))
        manifest = json.loads((pilot[0] / "pretrain_FullReco/dataset_manifest.json").read_text())
        batch = next(iter(runner._loader(manifest, config, "train")))
        if mode != "scratch":
            for key, tensor in source.state_dict().items():
                torch.testing.assert_close(model.encoder.state_dict()[key], tensor, rtol=0, atol=0)
        before = copy.deepcopy(model.encoder.state_dict())
        head_before = runner._initial_state_digest(model.head)
        model.train()
        if mode == "linear_probe":
            assert not model.encoder.training and not any(module.training for module in model.encoder.modules())
            assert model.head.training and not any(p.requires_grad for p in model.encoder.parameters())
            torch.testing.assert_close(model(batch), model(batch), rtol=0, atol=0)
        else:
            assert model.encoder.training and all(p.requires_grad for p in model.encoder.parameters())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.01)
        loss = torch.nn.functional.cross_entropy(model(batch), batch.targets)
        loss.backward()
        optimizer.step()
        changed = any(not torch.equal(tensor, model.encoder.state_dict()[key]) for key, tensor in before.items())
        assert changed == (mode != "linear_probe")
        assert runner._initial_state_digest(model.head) != head_before
        if mode == "linear_probe":
            assert all(p.grad is None for p in model.encoder.parameters())
        else:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
        model.eval().train()
        assert model.encoder.training == (mode != "linear_probe")
    assert len(set(heads)) == 1


@pytest.mark.parametrize("mismatch", ["view", "feature_order", "limits", "transform", "architecture", "keys"])
def test_reject_incompatible_encoder_contract_or_weights(pilot, tmp_path, mismatch):
    config = config_for(pilot, "linear_probe")
    if mismatch == "view":
        config["data"]["view"] = "L1T"
    elif mismatch == "limits":
        config["data"]["collection_limits"]["PUPPIPart"] += 1
    elif mismatch == "transform":
        config["data"]["eta_clip"] += 1
    elif mismatch == "architecture":
        config["model"]["num_latents"] += 1
    else:
        payload = torch.load(config["downstream"]["encoder_checkpoint"], weights_only=True)
        if mismatch == "feature_order":
            payload["contract"]["collections"][0]["feature_names"].reverse()
        else:
            payload["state_dict"].pop(next(iter(payload["state_dict"])))
        path = tmp_path / "incompatible.pt"
        torch.save(payload, path)
        config["downstream"]["encoder_checkpoint"] = str(path)
    with pytest.raises(ValueError, match="contract|strict key/shape"):
        runner._make_model(config, {"background": 0, "signal": 1})


def test_configuration_validation_and_legacy_head(pilot):
    # The editable main configs have different horizons (supervised 24/downstream 3).
    # Compare the resolved common protocol without resetting the user's defaults.
    common = {"epochs": 2}
    legacy = runner.load_run_config(REPO / "configs/runs/collide_supervised.yaml", common)
    assert "downstream" not in legacy
    torch.manual_seed(42)
    model = runner._make_model(legacy, {"background": 0, "signal": 1})
    assert isinstance(model.head, torch.nn.Sequential)
    assert not model.encoder.tokenizer.mask_indicators
    torch.manual_seed(42)
    restored = runner._make_model(legacy, {"background": 0, "signal": 1}, initialize_encoder=False)
    assert runner._initial_state_digest(model) == runner._initial_state_digest(restored)
    with pytest.raises(ValueError, match="requires encoder_checkpoint"):
        runner.load_run_config(pilot[1], {"downstream_mode": "linear_probe"})
    with pytest.raises(ValueError, match="scratch must not"):
        runner.load_run_config(pilot[1], {"encoder_checkpoint": "unused.pt"})
    with pytest.raises(ValueError, match="downstream.mode"):
        runner.load_run_config(pilot[1], {"downstream_mode": "unknown"})
    for view in ("FullReco", "L1T"):
        config = runner.load_run_config(REPO / "configs/runs/collide_downstream.yaml", {**common, "view": view})
        assert config["model"] == legacy["model"]
        assert config["training"] == legacy["training"]
        assert config["partition_manifest"] == legacy["partition_manifest"]


@pytest.mark.parametrize("field,value", [("partition_sha256", "changed"), ("test_used", True), ("pretraining_split", "test")])
def test_inspection_rejects_incompatible_pretraining_provenance(pilot, tmp_path, field, value):
    config = config_for(pilot, "linear_probe", output=tmp_path / "bad")
    payload = torch.load(config["downstream"]["encoder_checkpoint"], weights_only=True)
    payload["provenance"][field] = value
    path = tmp_path / "bad_encoder.pt"
    torch.save(payload, path)
    config["downstream"]["encoder_checkpoint"] = str(path)
    with pytest.raises(ValueError, match="provenance"):
        runner.run_classification(config, inspect_only=True)
    assert not (tmp_path / "bad/checkpoint.pt").exists()


@pytest.fixture(scope="module")
def benchmark(pilot):
    root, _, partitions = pilot
    results, paths = {}, {}
    original_epoch = runner._run_epoch
    for view in ("FullReco", "L1T"):
        results[view], paths[view] = {}, []
        for mode in MODES:
            config = config_for(pilot, mode, view=view)
            output = Path(config["output_dir"])
            order = []
            def audited_epoch(model, loader, device, label_map, **kwargs):
                split = loader.dataset.split
                order.append((split, kwargs.get("optimizer") is not None))
                if split == "test":
                    selected = torch.load(output / "checkpoint.pt", weights_only=True)
                    assert kwargs.get("optimizer") is None
                    assert all(torch.equal(v.cpu(), selected["model_state_dict"][k]) for k, v in model.state_dict().items())
                    assert sum(s == "validation" for s, _ in order) == config["training"]["epochs"] + 1
                return original_epoch(model, loader, device, label_map, **kwargs)
            runner._run_epoch = audited_epoch
            try:
                result = runner.run_classification(config)
            finally:
                runner._run_epoch = original_epoch
            assert order[-1] == ("test", False) and sum(split == "test" for split, _ in order) == 1
            assert all(split == "train" for split, optimizing in order if optimizing)
            assert result["best_epoch"] == min(result["history"], key=lambda e: e["validation"]["loss"])["epoch"]
            assert result["downstream"]["parameters"]["trainable"] > 0
            assert result["performance"]["train_events"] == 48
            assert result["performance"]["train_events_per_second"] > 0
            for split in ("validation", "test"):
                _, ids, labels, _ = read_evaluation_scores(output / result[split]["score_manifest"])
                expected = {e["id"]: process["label"] for process in partitions["processes"] for e in process["events"] if e["split"] == split}
                assert dict(zip(ids, labels.tolist())) == expected
            results[view][mode] = result
            paths[view].append(output)
    return root, results, paths


def test_cpu_end_to_end_comparison_and_self_contained_evaluation(benchmark, pilot, tmp_path):
    root, results, paths = benchmark
    for view in ("FullReco", "L1T"):
        compared = compare_downstream_runs(paths[view], root / f"comparison_{view}")
        assert len(compared["results"]) == 6
        assert {r["mode"] for r in compared["results"]} == set(MODES)
        assert (root / f"comparison_{view}/comparison.html").is_file()
        for plot in ("roc_validation", "roc_test", "learning_curves"):
            assert (root / f"comparison_{view}/plots/{plot}.png").stat().st_size > 1000
            assert (root / f"comparison_{view}/plots/{plot}.pdf").stat().st_size > 1000
        with pytest.raises(FileExistsError, match="overwrite"):
            compare_downstream_runs(paths[view], root / f"comparison_{view}")
    # Evaluation uses weights/config inside the classifier checkpoint, even if the source export moves.
    original = root / "pretrain_FullReco/encoder.pt"
    relocated = root / "pretrain_FullReco/moved_encoder.pt"
    original.rename(relocated)
    try:
        for mode in ("linear_probe", "finetune"):
            directory = root / f"FullReco_{mode}"
            evaluation = runner.evaluate_run(directory / "checkpoint.pt", output_dir=tmp_path / mode, device="cpu", batch_size=5)
            assert evaluation["downstream"] == results["FullReco"][mode]["downstream"]
            assert evaluation["test"]["loss"] == pytest.approx(results["FullReco"][mode]["test"]["loss"], abs=1e-6)
            assert evaluation["test"]["binary"]["roc_auc"] == results["FullReco"][mode]["test"]["binary"]["roc_auc"]
        compare_downstream_runs([paths["FullReco"][0], tmp_path / "linear_probe", tmp_path / "finetune"], tmp_path / "reevaluated_comparison")
    finally:
        relocated.rename(original)
    with pytest.raises(ValueError, match="random initialization"):
        compare_classification_runs([paths["FullReco"][1], paths["L1T"][1]], tmp_path / "wrong_comparator")
    command = [sys.executable, str(REPO / "scripts/run_classification.py"), "--stage", "compare-transfer", "--output-dir", str(tmp_path / "cli_compare")]
    for directory in paths["FullReco"]:
        command += ["--run-dir", str(directory)]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_comparison_rejects_inconsistent_protocol_scores_and_checkpoint(benchmark, tmp_path):
    root, _, paths = benchmark
    directory = paths["FullReco"][1]
    metrics_path = directory / "metrics.json"
    original = metrics_path.read_text()
    for key in ("training_protocol", "input_contract", "downstream", "best_validation_loss"):
        metrics = json.loads(original)
        if key == "training_protocol":
            metrics[key]["training"]["learning_rate"] *= 2
        elif key == "input_contract":
            metrics[key]["collections"][0]["feature_names"].reverse()
        elif key == "downstream":
            metrics[key]["initial_head_sha256"] = "changed"
        else:
            metrics[key] += 1
        metrics_path.write_text(json.dumps(metrics))
        try:
            with pytest.raises(ValueError, match="mismatch"):
                compare_downstream_runs(paths["FullReco"], tmp_path / key)
            assert not (tmp_path / key).exists()
        finally:
            metrics_path.write_text(original)
    with pytest.raises(ValueError, match="view differs"):
        compare_downstream_runs([paths["FullReco"][0], paths["FullReco"][1], paths["L1T"][2]], tmp_path / "mixed_views")
    # Corrupt a scored identity while updating its shard hash: partition checks must still reject it.
    score_manifest = directory / "scores/test/manifest.json"
    saved_manifest = score_manifest.read_text()
    info = json.loads(saved_manifest)
    shard = score_manifest.parent / info["shards"][0]["path"]
    original_bytes = shard.read_bytes()
    table = pq.read_table(shard)
    ids = table["event_id"].to_pylist()
    ids[0] = "origin-v1:" + "a" * 64
    table = table.set_column(0, table.schema.field(0), pa.array(ids))
    pq.write_table(table, shard)
    info["shards"][0]["sha256"] = sha256_file(shard)
    score_manifest.write_text(json.dumps(info))
    try:
        with pytest.raises(ValueError, match="events/labels/context"):
            compare_downstream_runs(paths["FullReco"], tmp_path / "wrong_ids")
    finally:
        shard.write_bytes(original_bytes)
        score_manifest.write_text(saved_manifest)
    checkpoint = directory / "checkpoint.pt"
    saved_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(saved_bytes + b"changed")
    try:
        with pytest.raises(ValueError, match="fingerprint"):
            compare_downstream_runs(paths["FullReco"], tmp_path / "wrong_checkpoint")
    finally:
        checkpoint.write_bytes(saved_bytes)


def test_cli_modes_inspection_and_protection(pilot, tmp_path):
    config = config_for(pilot, "linear_probe")
    command = [sys.executable, str(REPO / "scripts/run_classification.py"), "--config", str(pilot[1]),
               "--stage", "inspect", "--downstream-mode", "linear_probe", "--encoder-checkpoint",
               config["downstream"]["encoder_checkpoint"], "--output-dir", str(tmp_path / "inspect")]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "inspect/checkpoint.pt").exists()
    command[command.index("inspect")] = "train"
    completed = subprocess.run(command + ["--epochs", "1"], cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    repeated = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert repeated.returncode == 1 and "overwrite" in repeated.stderr


@pytest.mark.parametrize("difference", ["seed", "signal", "encoder"])
def test_comparison_rejects_independently_valid_but_different_experiments(benchmark, pilot, tmp_path, difference):
    _, _, paths = benchmark
    config = config_for(pilot, "finetune", output=tmp_path / "different_run")
    expected = "training_protocol"
    if difference == "seed":
        config["seed"] += 1
    elif difference == "signal":
        config["evaluation"]["signal_process"] = "background"
        expected = "signal_process"
    else:
        source = torch.load(config["downstream"]["encoder_checkpoint"], weights_only=True)
        first = next(iter(source["state_dict"]))
        source["state_dict"][first] = source["state_dict"][first] + .1
        alternate = tmp_path / "different_encoder.pt"
        torch.save(source, alternate)
        config["downstream"]["encoder_checkpoint"] = str(alternate)
        expected = "same pretrained encoder"
    runner.run_classification(config)
    with pytest.raises(ValueError, match=expected):
        compare_downstream_runs([paths["FullReco"][0], paths["FullReco"][1], Path(config["output_dir"])], tmp_path / "invalid_comparison")
    assert not (tmp_path / "invalid_comparison").exists()


def test_prepared_fast_and_strict_full_pipeline_share_scientific_memberships(tmp_path, monkeypatch):
    """Both views × both policies × three modes; one tiny CPU epoch, synthetic data."""
    from flowyforge.data_plugins.collide_v2.sample_preparation import load_sample_config, prepare_sample
    from flowyforge.data_plugins.collide_v2.streaming import load_partition_manifest
    from flowyforge.training import pretraining_run
    from flowyforge.data_plugins.collide_v2 import streaming
    make_tiny_parquet_sample(tmp_path / "raw", events_per_class=16, files_per_class=2)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"model": {"embedding_dim": 8, "num_latents": 2, "num_heads": 2,
        "num_layers": 1, "output_dim": 8, "dropout": 0.}, "data": {"collection_limits": {
        "PUPPIPart": 8, "JetPuppiAK4": 3, "Electron": 2, "MuonTight": 2, "PhotonTight": 2, "PUPPIMET": 1}}}))
    memberships = []
    manifest_hashes = []
    real_loader = pretraining_run._loader
    def train_validation_only(manifest, config, split, **kwargs):
        assert split in ("train", "validation")
        assert kwargs["purpose"] == ("pretraining" if split == "train" else "checkpoint_selection")
        return real_loader(manifest, config, split, **kwargs)
    monkeypatch.setattr(pretraining_run, "_loader", train_validation_only)
    real_epoch = runner._run_epoch
    observed = []
    def train_then_final_test(model, loader, *args, **kwargs):
        split = loader.dataset.split
        observed.append((split, kwargs.get("optimizer") is not None))
        if split == "test":
            assert not kwargs.get("optimizer")
            assert sum(s == "validation" for s, _ in observed) == 2
        return real_epoch(model, loader, *args, **kwargs)
    monkeypatch.setattr(runner, "_run_epoch", train_then_final_test)
    for policy in ("fast", "strict"):
        directory = tmp_path / policy
        sample_path = tmp_path / f"sample_{policy}.yaml"
        sample_path.write_text(yaml.safe_dump({"source": {"kind": "local", "processes": {
            n: str(tmp_path / "raw" / n) for n in ("background", "signal")}},
            "output_dir": str(directory / "data"), "model_config": str(model_path),
            "selection": {"events_per_process": 16, "seed": 42, "read_batch_size": 8},
            "partitions": {"check_integrity_SHA256": policy, "num_workers": 1}}))
        prepare_sample(load_sample_config(sample_path))
        parts_path = directory / "data/partitions.json"
        parts = load_partition_manifest(parts_path)
        if policy == "strict":
            assert create_partition_manifest({"root": str(directory / "data")}) == parts
        memberships.append({(p["name"], p["label"], e["id"], e["split"]) for p in parts["processes"] for e in p["events"]})
        manifest_hashes.append(parts["sha256"])
        raw = {"model_config": str(model_path), "partition_manifest": str(parts_path), "seed": 42,
               "sources": {"small": {"processes": {n: str(directory / "data" / n) for n in ("background", "signal")}, "max_events_per_class": None}},
               "data": {"read_batch_size": 8, "shuffle_buffer": 4, "audit_events_per_class": 2},
               "training": {"epochs": 1, "batch_size": 8, "device": "cpu", "cpu_threads": 1},
               "evaluation": {"signal_process": "signal", "background_acceptance": .5,
                              "min_background_events": 1, "score_shard_rows": 4}}
        downstream_path = directory / "run.yaml"
        downstream_path.write_text(yaml.safe_dump(raw))
        pretrain_raw = {**raw, "pretraining": {"probability": .5, "decoder_hidden_dim": 8},
                        "reporting": {"inference": False}}
        pretrain_path = directory / "pretrain.yaml"
        pretrain_path.write_text(yaml.safe_dump(pretrain_raw))
        with monkeypatch.context() as guards:
            if policy == "fast":
                def no_data_hash(path):
                    assert not Path(path).resolve().is_relative_to((directory / "data").resolve())
                    return sha256_file(path)
                guards.setattr(streaming, "sha256_file", no_data_hash)
                guards.setattr(runner, "sha256_file", no_data_hash)
            for view in ("FullReco", "L1T"):
                encoder_path = directory / f"pretrain_{view}/encoder.pt"
                pre = load_pretraining_config(pretrain_path, {"view": view, "output_dir": str(encoder_path.parent)})
                pre_metrics = run_pretraining(pre)
                assert pre_metrics["test_used"] is False
                payload = torch.load(encoder_path, weights_only=True)
                assert payload["provenance"]["partition_sha256"] == parts["sha256"]
                assert payload["provenance"]["pretraining_split"] == "train"
                outputs = []
                for mode in MODES:
                    observed.clear()
                    output = directory / f"{view}_{mode}"
                    overrides = {"view": view, "downstream_mode": mode, "output_dir": str(output)}
                    if mode != "scratch":
                        overrides["encoder_checkpoint"] = str(encoder_path)
                    cfg = runner.load_run_config(downstream_path, overrides)
                    metrics = runner.run_classification(cfg)
                    assert observed[-1] == ("test", False) and sum(s == "test" for s, _ in observed) == 1
                    assert all(s == "train" for s, optimize in observed if optimize)
                    for split in ("validation", "test"):
                        _, ids, labels, _ = read_evaluation_scores(output / metrics[split]["score_manifest"])
                        assert dict(zip(ids, labels.tolist())) == {e["id"]: p["label"] for p in parts["processes"]
                            for e in p["events"] if e["split"] == split}
                    outputs.append(output)
                compare_downstream_runs(outputs, directory / f"comparison_{view}")
                assert (directory / f"comparison_{view}/comparison.html").is_file()
                if policy == "fast" and view == "FullReco":
                    observed.clear()
                    resume_config = runner.load_run_config(downstream_path, {"view": view,
                        "downstream_mode": "linear_probe", "encoder_checkpoint": str(encoder_path),
                        "output_dir": str(outputs[1]), "epochs": 2})
                    resumed = runner.run_classification(resume_config, resume=outputs[1] / "last_checkpoint.pt")
                    assert len(resumed["history"]) == 2
                    saved = torch.load(outputs[1] / "last_checkpoint.pt", weights_only=True)
                    assert saved["source_sha256"] == {}  # No invented full-data checksums.
        if policy == "strict":
            # Equal memberships do not authorize silently substituting the manifest SHA in encoder provenance.
            cfg = runner.load_run_config(downstream_path, {"downstream_mode": "linear_probe", "view": "FullReco",
                "encoder_checkpoint": str(tmp_path / "fast/pretrain_FullReco/encoder.pt"),
                "output_dir": str(directory / "incompatible_policy")})
            with pytest.raises(ValueError, match="provenance"):
                runner.run_classification(cfg, inspect_only=True)
    assert memberships[0] == memberships[1] and manifest_hashes[0] != manifest_hashes[1]
