"""Offline masking contracts, strict encoder transfer and tiny CPU resume."""

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

from flowyforge.data_plugins.collide_v2.object_dataset import (
    collate_event_batches, default_collection_specs, row_to_event_batch,
)
from flowyforge.data_plugins.collide_v2.streaming import create_partition_manifest, save_partition_manifest, StreamingDataset
from flowyforge.model_plugins.foundation_encoder import (
    EventClassificationModel, FoundationEncoder, MaskedReconstructionModel, encoder_contract, export_encoder, load_encoder,
)
from flowyforge.model_plugins.perceiver import PerceiverConfig
from flowyforge.task_plugins.masked_modeling import MaskedModelingTask
from flowyforge.training.classification_run import batch_to_device
from flowyforge.training.pretraining_run import _epoch, load_pretraining_config, run_pretraining

REPO = Path(__file__).resolve().parents[1]
LIMITS = {"PUPPIPart": 8, "JetPuppiAK4": 3, "Electron": 0, "MuonTight": 0, "PhotonTight": 0, "PUPPIMET": 0}
SPECS = default_collection_specs(LIMITS)
FEATURES = {s.name: list(s.feature_names) for s in SPECS}
MODEL = {"embedding_dim": 8, "num_heads": 2, "num_latents": 2, "num_layers": 1, "output_dim": 8, "dropout": 0.1}
DATA = {"view": "FullReco", "units": {"momentum": "GeV", "angle": "rad"},
        "collection_limits": LIMITS, "eta_clip": 5.0, "value_clip": 1.e6}


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def row(index=0, count=3, view="FullReco"):
    fields = {"PUPPIPart": {"PT": [float(10 + index + i) for i in range(count)], "Eta": [0.1] * count,
                             "Phi": [0.2] * count, "PID": [211.] * count, "Charge": [1.] * count},
              "JetPuppiAK4": {"PT": [float(30 + index)] if count else [], "Eta": [0.3] if count else [],
                              "Phi": [0.4] if count else [], "Mass": [5.] if count else []}}
    return {f"{view}_{name}_{field}": values for name, fs in fields.items() for field, values in fs.items()}


def event(index=0, count=3, raw=None):
    result = row_to_event_batch(raw if raw is not None else row(index, count), specs=SPECS, target=index % 2)
    result.metadata["event_id"] = f"origin-v1:{index:064x}"
    return result


def model():
    encoder = FoundationEncoder({k: len(v) for k, v in FEATURES.items()}, PerceiverConfig(**MODEL), mask_indicators=True)
    return MaskedReconstructionModel(encoder, MaskedModelingTask(FEATURES).targets, hidden_dim=8)


def test_original_validity_survives_sort_padding_collation_device_and_masking():
    raw = row(count=5)
    raw["FullReco_PUPPIPart_PT"] = [float("nan"), 5., 0., -1., float("inf")]
    del raw["FullReco_JetPuppiAK4_Mass"]
    batch = collate_event_batches([event(raw=raw), event(1, count=1), event(2, count=0)])
    original = copy.deepcopy(batch)
    task = MaskedModelingTask(FEATURES, probability=1)
    prepared = task.prepare_batch(batch, validation=True)
    assert prepared.inputs.targets is None
    part = batch.objects["PUPPIPart"]
    assert part.feature_valid[0, :, 0].tolist() == [True, False, True, False, False]
    assert part.metadata["events"][0]["object_indices"] == [1, 0, 2, 3, 4]
    assert prepared.masks["PUPPIPart"][0, :, 0].sum() == 2  # real zero is a target
    assert not prepared.masks["JetPuppiAK4"][0, :, 1].any()  # missing Mass is not
    for name, group in batch.objects.items():
        torch.testing.assert_close(group.features, original.objects[name].features)
        assert group.feature_mask is None
        hidden = prepared.inputs.objects[name].feature_mask
        assert not hidden[~group.mask].any()
        assert not hidden[~group.feature_valid].any()
        assert not prepared.inputs.objects[name].features[hidden].any()
    moved = batch_to_device(prepared.inputs, torch.device("cpu"))
    assert moved.objects["PUPPIPart"].feature_mask.dtype == torch.bool
    assert moved.objects["PUPPIPart"].feature_valid.dtype == torch.bool
    # Indicators also survive collation when corruption occurs before collation.
    singles = [task.prepare_batch(event(i), validation=True).inputs for i in range(2)]
    assert collate_event_batches(singles).objects["PUPPIPart"].feature_mask[..., 0].all()
    invalid = row()
    invalid["FullReco_JetPuppiAK4_Mass"] = [float("nan")]
    assert not event(raw=invalid).objects["JetPuppiAK4"].feature_valid[..., -1].any()


def test_no_target_leak_tokenize_once_and_finite_gradients_through_global_embedding():
    batch = collate_event_batches([event(0), event(1, 1), event(2, 0)])
    task = MaskedModelingTask(FEATURES, probability=1)
    prepared = task.prepare_batch(batch)
    network = model().eval()
    calls = []
    hook = network.encoder.tokenizer.register_forward_hook(lambda *args: calls.append(1))
    predictions = network(prepared.inputs)
    hook.remove()
    assert len(calls) == 1
    poisoned = copy.deepcopy(prepared.inputs)
    for group in poisoned.objects.values():
        group.features[group.feature_mask] = float("nan")
    again = network(poisoned)
    for name in predictions:
        torch.testing.assert_close(predictions[name], again[name], rtol=0, atol=0)
    poisoned_targets = copy.deepcopy(prepared)
    for name, targets in poisoned_targets.targets.items():
        targets[~prepared.masks[name]] = float("nan")
    loss = task.compute_loss(predictions, prepared)
    torch.testing.assert_close(loss, task.compute_loss(predictions, poisoned_targets))
    loss.backward()
    for component in (network.encoder.tokenizer, network.encoder.encoder, network.encoder.encoder.output, network.decoders):
        gradients = [p.grad for p in component.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    for name in prepared.targets:
        # Padding has neither output loss nor any hidden indicator.
        assert not prepared.masks[name][~batch.objects[name].mask].any()
    tokens = network.encoder.forward_features(prepared.inputs).tokens
    assert list(tokens.group_slices) == list(FEATURES)
    assert not tokens.tokens[~tokens.mask].any()
    zeros = collate_event_batches([event(raw=row(index=-10, count=1))])
    masked = task.prepare_batch(zeros).inputs
    plain_tokens = network.encoder.tokenizer(zeros).tokens
    masked_tokens = network.encoder.tokenizer(masked).tokens
    assert not torch.equal(plain_tokens, masked_tokens)  # learned indicator != genuine zero


def test_empty_batch_no_eligible_targets_and_balanced_feature_collection_loss():
    task = MaskedModelingTask(FEATURES, probability=1, collection_weights={"PUPPIPart": 2},
                             feature_weights={"JetPuppiAK4/log1p_Mass": 3})
    empty = task.prepare_batch(collate_event_batches([event(count=0)]))
    network = model().eval()
    assert torch.isfinite(network.encoder.encode(empty.inputs)).all()
    assert task.compute_loss(network(empty.inputs), empty) is None
    sums = {"PUPPIPart/log1p_PT": (12., 3), "JetPuppiAK4/log1p_PT": (2., 2), "JetPuppiAK4/log1p_Mass": (10., 2)}
    assert task.combine(sums) == pytest.approx((2 * 4 + (1 + 3 * 5) / 4) / 3)
    batch = collate_event_batches([event()])
    for group in batch.objects.values():
        group.feature_valid.zero_()
    prepared = task.prepare_batch(batch)
    assert task.compute_loss(network(prepared.inputs), prepared) is None
    batch.objects["PUPPIPart"].feature_valid = None
    with pytest.raises(ValueError, match="feature_valid"):
        task.prepare_batch(batch)


def test_validation_masks_independent_of_batch_order_padding_and_epoch():
    events = [event(i, 1 + i % 5) for i in range(15)]
    task = MaskedModelingTask(FEATURES, probability=.5)
    together = task.prepare_batch(collate_event_batches(events), validation=True, epoch=1)
    for i, single in reversed(list(enumerate(events))):
        separately = task.prepare_batch(collate_event_batches([single]), validation=True, epoch=99)
        for name in together.masks:
            n = separately.masks[name].shape[1]
            assert torch.equal(together.masks[name][i, :n], separately.masks[name][0])
    first = task.prepare_batch(collate_event_batches(events), epoch=1)
    second = task.prepare_batch(collate_event_batches(events), epoch=2)
    assert any(not torch.equal(first.masks[k], second.masks[k]) for k in first.masks)
    changed_labels = copy.deepcopy(events)
    for single in changed_labels:
        single.targets.fill_(99)
    assert torch.equal(together.masks["PUPPIPart"], task.prepare_batch(collate_event_batches(changed_labels), validation=True).masks["PUPPIPart"])


def test_no_mask_batch_skips_optimizer_and_validation_loss_ignores_batch_size():
    task = MaskedModelingTask(FEATURES, probability=.5)
    network = model()
    optimizer = torch.optim.AdamW(network.parameters(), weight_decay=.5)
    before = copy.deepcopy(network.state_dict())
    empty = collate_event_batches([event(count=0)])
    result = _epoch(network, [empty], task, torch.device("cpu"), epoch=1,
                    baseline={}, optimizer=optimizer)
    assert result["loss"] is None and result["optimizer_steps"] == 0
    assert result["skipped_batches"] == 1 and result["empty_events"] == 1
    for key, tensor in network.state_dict().items():
        torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)
    events = [event(i, 1 + i % 4) for i in range(10)]
    baseline = {"features": {task.key(n, f): {"value": 2.0}
                             for n, fs in task.targets.items() for f in fs}}
    together = _epoch(network, [collate_event_batches(events)], task, torch.device("cpu"), epoch=1, baseline=baseline)
    separate = _epoch(network, [collate_event_batches([e]) for e in reversed(events)], task,
                      torch.device("cpu"), epoch=99, baseline=baseline)
    assert together["loss"] == pytest.approx(separate["loss"], rel=1e-6)
    for key in together["features"]:
        assert together["features"][key]["masked_count"] == separate["features"][key]["masked_count"]
    assert together["training_constant"]["loss"] == pytest.approx(separate["training_constant"]["loss"])


def test_strict_encoder_export_transfer_and_eval_equivalence(tmp_path):
    network = model().eval()
    contract = encoder_contract(MODEL, DATA)
    path = tmp_path / "encoder.pt"
    export_encoder(path, network.encoder, contract, {"partition_sha256": "synthetic"})
    loaded = load_encoder(path, expected_contract=contract).eval()
    batch = collate_event_batches([event(), event(1, 0)])
    torch.testing.assert_close(network.encoder.encode(batch), loaded.encode(batch), rtol=0, atol=0)
    classifier = EventClassificationModel.from_encoder_checkpoint(path, 2, expected_contract=contract).eval()
    classifier.encoder.load_state_dict(network.encoder.state_dict(), strict=True)
    assert classifier(batch).shape == (2, 2)
    assert torch.isfinite(classifier(batch)).all()
    bad = copy.deepcopy(contract)
    bad["reader"]["view"] = "L1T"
    with pytest.raises(ValueError, match="contract"):
        load_encoder(path, expected_contract=bad)
    payload = torch.load(path, weights_only=True)
    payload["state_dict"].pop(next(iter(payload["state_dict"])))
    torch.save(payload, path)
    with pytest.raises(ValueError, match="strict key/shape"):
        load_encoder(path)


def make_run_config(tmp_path):
    processes = {}
    for label, name in enumerate(("ttbar", "Wjets")):
        path = tmp_path / f"{name}.parquet"
        rows = [{**row(i + label * 20, view="FullReco"), **row(i + label * 20, view="L1T")} for i in range(20)]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=5)
        processes[name] = str(path)
    partition_path = tmp_path / "partitions.json"
    partitions = create_partition_manifest({"processes": processes})
    save_partition_manifest(partitions, partition_path)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump({"model": MODEL, "data": {"collection_limits": LIMITS}}))
    config_path = tmp_path / "pretraining.yaml"
    config_path.write_text(yaml.safe_dump({"model_config": str(model_path), "partition_manifest": str(partition_path),
        "output_dir": str(tmp_path / "run"), "sources": {"small": {"processes": processes, "max_events_per_class": None}},
        "data": {"read_batch_size": 3, "shuffle_buffer": 4}, "pretraining": {"probability": .5, "decoder_hidden_dim": 8},
        "training": {"epochs": 1, "device": "cpu", "cpu_threads": 1, "batch_size": 8}}))
    return config_path, partitions


def test_tiny_training_resume_constant_train_only_and_no_test_access(tmp_path, monkeypatch):
    config_path, partitions = make_run_config(tmp_path)
    seen = []
    original_init = StreamingDataset.__init__
    def audited_init(self, manifest, split, **kwargs):
        assert split != "test"
        seen.append((split, kwargs.get("purpose")))
        original_init(self, manifest, split, **kwargs)
    monkeypatch.setattr(StreamingDataset, "__init__", audited_init)
    config = load_pretraining_config(config_path)
    output = Path(config["output_dir"])
    inspection = run_pretraining(config, inspect_only=True)
    assert inspection["validation_constant"]["loss"] is not None
    assert not (output / "checkpoint.pt").exists()
    result = run_pretraining(config)
    assert result["test_used"] is False and "test" not in result
    assert result["history"][0]["train"]["num_events"] == 24
    assert result["validation"]["num_events"] == 8
    assert result["constant_baseline"]["features"]["PUPPIPart/log1p_PT"]["count"] == 24 * 3
    assert result["validation"]["training_constant"]["loss"] >= 0
    # Independently estimate the constant using original rows of TRAIN only.
    values = []
    for process in partitions["processes"]:
        rows = pq.read_table(tmp_path / f"{process['name']}.parquet").to_pylist()
        for entry in process["events"]:
            if entry["split"] == "train":
                values.extend(row_to_event_batch(rows[entry["origin"]["row"]], specs=SPECS).objects["PUPPIPart"].features[0, :, 0].tolist())
    assert result["constant_baseline"]["features"]["PUPPIPart/log1p_PT"]["value"] == pytest.approx(sum(values) / len(values))
    with pytest.raises(FileExistsError, match="overwrite"):
        run_pretraining(config)
    extended = load_pretraining_config(config_path, {"epochs": 2})
    resumed = run_pretraining(extended, resume=output / "checkpoint.pt")
    full_config = load_pretraining_config(config_path, {"epochs": 2, "output_dir": str(tmp_path / "uninterrupted")})
    full = run_pretraining(full_config)
    assert resumed["history"] == full["history"]
    state = torch.load(output / "checkpoint.pt", weights_only=True)
    other = torch.load(tmp_path / "uninterrupted/checkpoint.pt", weights_only=True)
    for key in state["model_state_dict"]:
        torch.testing.assert_close(state["model_state_dict"][key], other["model_state_dict"][key], rtol=0, atol=0)
    exported = load_encoder(output / "encoder.pt").eval()
    for key, tensor in exported.state_dict().items():
        torch.testing.assert_close(tensor, state["best_encoder_state_dict"][key], rtol=0, atol=0)
    assert resumed["best_epoch"] == min(resumed["history"], key=lambda e: e["validation"]["loss"])["epoch"]
    assert set(seen) == {("train", "pretraining"), ("validation", "checkpoint_selection")}
    bad = load_pretraining_config(config_path, {"epochs": 3, "view": "L1T"})
    with pytest.raises(ValueError, match="resume configuration"):
        run_pretraining(bad, resume=output / "checkpoint.pt")
    # The same immutable event partitions support an independent L1T encoder.
    l1t = load_pretraining_config(config_path, {"view": "L1T", "output_dir": str(tmp_path / "l1t")})
    result_l1t = run_pretraining(l1t)
    assert result_l1t["partition_sha256"] == result["partition_sha256"]
    assert load_encoder(tmp_path / "l1t/encoder.pt").input_contract["reader"]["view"] == "L1T"


def test_pretraining_cli_inspect_and_configuration(tmp_path):
    config_path, _ = make_run_config(tmp_path)
    command = [sys.executable, str(REPO / "scripts/run_pretraining.py"), "--config", str(config_path), "--stage", "inspect"]
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "test excluded" in completed.stdout
    config = load_pretraining_config(REPO / "configs/runs/collide_pretraining.yaml")
    assert config["pretraining"]["probability"] == .15
    assert "PUPPIMET" not in config["pretraining"]["targets"]
    raw = yaml.safe_load(config_path.read_text())
    raw.pop("partition_manifest")
    config_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="shared partition_manifest"):
        load_pretraining_config(config_path)
