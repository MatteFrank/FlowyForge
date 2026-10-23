"""Offline contract and end-to-end tests using only synthetic local Parquet."""

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from flowyforge.core.typing import EventBatch, ObjectGroup
from flowyforge.data_plugins.collide_v2.object_dataset import (
    CollideCollectionSpec, collate_event_batches, collection_feature_dims,
    default_collection_specs, expand_parquet_paths, iter_parquet_events,
    load_process_events, required_columns, row_to_event_batch,
)
from flowyforge.model_plugins import (
    EventClassificationModel, FoundationEncoder, PerceiverConfig, PerceiverEventEncoder,
)
from flowyforge.model_plugins.base import ModelPlugin
from flowyforge.representation.tokenizer import ObjectTokenizer, TokenBatch
from flowyforge.training.foundation_smoke import stratified_event_split


@pytest.fixture(scope="module", autouse=True)
def single_thread_torch():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def synthetic_row(view="FullReco", count=3, offset=0.0, wrapped=False):
    # Explicit schema fixture, deliberately independent of the reader specs.
    collections = {
        "PUPPIPart": {"PT": [10.0 + offset, 30.0 + offset, 20.0 + offset][:count],
                      "Eta": [0.0, 10.0, -10.0][:count], "Phi": [0.0, math.pi / 2, math.pi][:count],
                      "PID": [22, 211, -211][:count], "Charge": [0, 1, -1][:count]},
        "JetPuppiAK4": {"PT": [50.0 + offset][:min(count, 1)], "Eta": [0.5][:min(count, 1)],
                        "Phi": [0.25][:min(count, 1)], "Mass": [5.0][:min(count, 1)]},
        "Electron": {"PT": [12.0][:min(count, 1)], "Eta": [-0.1][:min(count, 1)],
                     "Phi": [1.0][:min(count, 1)], "Charge": [-1][:min(count, 1)]},
        "MuonTight": {"PT": [15.0][:min(count, 1)], "Eta": [0.2][:min(count, 1)],
                      "Phi": [-0.5][:min(count, 1)], "Charge": [1][:min(count, 1)]},
        "PhotonTight": {"PT": [22.0][:min(count, 1)], "Eta": [0.3][:min(count, 1)],
                        "Phi": [2.0][:min(count, 1)]},
        "PUPPIMET": {"MET": [40.0 + offset][:min(count, 1)], "Phi": [-2.0][:min(count, 1)]},
    }
    return {f"{view}_{name}_{field}": [values] if wrapped else values
            for name, fields in collections.items() for field, values in fields.items()}


@pytest.mark.parametrize("view", ["FullReco", "L1T"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_row_transform_sort_truncate_and_metadata(view, wrapped):
    specs = default_collection_specs({"PUPPIPart": 2})
    event = row_to_event_batch(synthetic_row(view, wrapped=wrapped), view, specs, target=1)
    assert event.targets.tolist() == [1]
    assert event.metadata["view"] == view
    assert collection_feature_dims(specs) == {
        "PUPPIPart": 6, "JetPuppiAK4": 5, "Electron": 5,
        "MuonTight": 5, "PhotonTight": 4, "PUPPIMET": 3,
    }
    group = event.objects["PUPPIPart"]
    assert group.features.shape == (1, 2, 6)
    assert group.features.dtype == torch.float32
    assert group.mask.tolist() == [[True, True]]
    assert group.metadata["original_count"] == 3
    assert group.metadata["truncated_count"] == 1
    assert group.metadata["source_columns"][0] == f"{view}_PUPPIPart_PT"
    torch.testing.assert_close(group.features[0, :, 0], torch.log1p(torch.tensor([30.0, 20.0])))
    torch.testing.assert_close(group.features[0, :, 1], torch.tensor([1.0, -1.0]))
    torch.testing.assert_close(group.features[0, :, 2], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(group.features[0, :, 3], torch.tensor([0.0, -1.0]), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(group.features[0, :, 4], torch.tensor([math.log1p(211), -math.log1p(211)]))
    assert group.features[0, :, 5].tolist() == [1.0, -1.0]
    assert event.objects["JetPuppiAK4"].features[0, 0, 4].item() == pytest.approx(math.log1p(5))
    assert event.objects["PUPPIMET"].features[0, 0, 0].item() == pytest.approx(math.log1p(40))


def test_nonfinite_optional_fields_and_clipping():
    row = synthetic_row()
    row["FullReco_PUPPIPart_PT"] = [float("nan"), float("inf"), -10.0]
    row["FullReco_PUPPIPart_Eta"] = [float("-inf"), 1e100, None]
    row["FullReco_PUPPIPart_Phi"] = [float("nan"), 1e100, float("inf")]
    del row["FullReco_PUPPIPart_PID"]
    row["FullReco_JetPuppiAK4_Mass"] = [1e100]
    row["FullReco_PUPPIMET_MET"] = [-5.0]
    event = row_to_event_batch(row)
    assert all(torch.isfinite(group.features).all() for group in event.objects.values())
    assert event.objects["PUPPIPart"].features[0, :, 0].tolist() == [0, 0, 0]
    assert event.objects["PUPPIPart"].features[0, :, 4].tolist() == [0, 0, 0]
    assert event.objects["PUPPIPart"].metadata["nonfinite_values"] == 6
    assert event.objects["PUPPIPart"].metadata["missing_optional_fields"] == ["PID"]
    assert event.objects["JetPuppiAK4"].features[0, 0, 4].item() == pytest.approx(math.log1p(1e6))
    assert event.objects["PUPPIMET"].features[0, 0, 0].item() == 0


def test_padding_masks_collation_and_variable_collections():
    full = row_to_event_batch(synthetic_row(), target=0)
    short = row_to_event_batch(synthetic_row(count=1), target=1)
    empty = row_to_event_batch(synthetic_row(count=0), target=0)
    del short.objects["Electron"]
    batch = collate_event_batches([full, short, empty])
    group = batch.objects["PUPPIPart"]
    assert group.features.shape == (3, 3, 6)
    assert group.mask.tolist() == [[True, True, True], [True, False, False], [False, False, False]]
    assert torch.count_nonzero(group.features[~group.mask]) == 0
    assert batch.objects["Electron"].mask.tolist() == [[True], [False], [False]]
    assert batch.targets.tolist() == [0, 1, 0]
    assert len(group.metadata["events"]) == 3
    assert batch.metadata["batch_size"] == 3
    assert collate_event_batches([empty, empty]).objects["PUPPIPart"].features.shape == (2, 0, 6)
    null_row = {column: None for column in synthetic_row()}
    assert all(group.features.shape[1] == 0 for group in row_to_event_batch(null_row).objects.values())


@pytest.mark.parametrize("bad_value", [1.0, [[1.0, 2.0], [3.0]], ["bad"]])
def test_reader_rejects_ambiguous_or_scalar_values(bad_value):
    row = synthetic_row()
    row["FullReco_PUPPIPart_PT"] = bad_value
    with pytest.raises(ValueError, match="object|numeric"):
        row_to_event_batch(row)


def test_schema_errors_are_explicit_and_do_not_invent_aliases():
    row = synthetic_row()
    row["FullReco_JetAK4_PT"] = row.pop("FullReco_JetPuppiAK4_PT")
    with pytest.raises(ValueError, match="FullReco_JetPuppiAK4_PT"):
        row_to_event_batch(row)
    with pytest.raises(ValueError, match="Scalar-summary"):
        row_to_event_batch({"FullReco_PUPPIPart_PT_mean": 2.0})
    with pytest.raises(ValueError, match="view"):
        required_columns("offline")
    row = synthetic_row()
    row["FullReco_PUPPIPart_Charge"] = [1]
    with pytest.raises(ValueError, match="inconsistent field lengths"):
        row_to_event_batch(row)
    with pytest.raises(ValueError, match="eta_clip"):
        row_to_event_batch(synthetic_row(), eta_clip=0)
    with pytest.raises(ValueError, match="Unknown collection"):
        default_collection_specs({"JetAK4": 1})
    with pytest.raises(ValueError, match="nonnegative"):
        default_collection_specs({"PUPPIPart": -1})
    assert "PhotonTight" not in collection_feature_dims(default_collection_specs({"PhotonTight": 0}))
    spec = CollideCollectionSpec("PUPPIPart", ("PT", "Eta", "Phi"), sort_by_pt=False)
    event = row_to_event_batch(synthetic_row(), specs=[spec])
    assert event.objects["PUPPIPart"].features[0, 0, 0].item() == pytest.approx(math.log1p(10))


def test_collation_rejects_mixed_targets_views_and_feature_orders():
    event = row_to_event_batch(synthetic_row(), target=0)
    with pytest.raises(ValueError, match="empty"):
        collate_event_batches([])
    with pytest.raises(ValueError, match="labeled"):
        collate_event_batches([event, row_to_event_batch(synthetic_row())])
    with pytest.raises(ValueError, match="FullReco and L1T"):
        collate_event_batches([event, row_to_event_batch(synthetic_row("L1T"), "L1T", target=0)])
    other = copy.deepcopy(event)
    other.objects["PUPPIPart"].metadata["feature_names"].reverse()
    with pytest.raises(ValueError, match="ordering"):
        collate_event_batches([event, other])


def test_tokenizer_shapes_stable_types_and_validation():
    batch = collate_event_batches([row_to_event_batch(synthetic_row()), row_to_event_batch(synthetic_row(count=1))])
    tokenizer = ObjectTokenizer(collection_feature_dims(), embedding_dim=16)
    result = tokenizer.tokenize(batch)
    assert isinstance(result, TokenBatch)
    assert result.tokens.shape == (2, 8, 16)
    assert result.mask.shape == result.group_ids.shape == (2, 8)
    assert result.group_ids.dtype == torch.long
    assert result.group_ids[0].tolist() == [0, 0, 0, 1, 2, 3, 4, 5]
    assert result.mask.sum(1).tolist() == [8, 6]
    assert torch.count_nonzero(result.tokens[~result.mask]) == 0
    reordered = EventBatch(objects=dict(reversed(list(batch.objects.items()))))
    torch.testing.assert_close(result.tokens, tokenizer(reordered).tokens)
    subset = tokenizer(EventBatch(objects={"PUPPIMET": batch.objects["PUPPIMET"]}))
    assert subset.group_ids.tolist() == [[5], [5]]
    bad = copy.deepcopy(batch)
    bad.objects["PUPPIPart"].features = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="expected features"):
        tokenizer(bad)
    with pytest.raises(ValueError, match="Unconfigured"):
        tokenizer(EventBatch(objects={"unknown": batch.objects["Electron"]}))
    with pytest.raises(TypeError, match="EventBatch"):
        tokenizer({})
    bad = copy.deepcopy(batch)
    bad.objects["PUPPIPart"].features[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        tokenizer(bad)


@pytest.mark.parametrize("view", ["FullReco", "L1T"])
def test_forward_backward_finite_gradients_and_all_paths_learn(view):
    batch = collate_event_batches([
        row_to_event_batch(synthetic_row(view), view, target=0),
        row_to_event_batch(synthetic_row(view, offset=30), view, target=1),
        row_to_event_batch(synthetic_row(view, count=0), view, target=0),
    ])
    config = PerceiverConfig(embedding_dim=16, num_heads=4, num_latents=4, num_layers=2, output_dim=12, dropout=0)
    encoder = FoundationEncoder(collection_feature_dims(), config)
    model = EventClassificationModel(encoder, 2)
    assert isinstance(encoder, ModelPlugin) and isinstance(model, ModelPlugin)
    assert model.encoder is encoder
    assert encoder(batch).shape == (3, 12)
    torch.testing.assert_close(encoder.encode(batch), encoder.get_embeddings(batch))
    assert PerceiverEventEncoder(config)(encoder.tokenizer(batch)).shape == (3, 12)
    logits = model(batch, task="classification")
    assert logits.shape == (3, 2)
    loss = torch.nn.functional.cross_entropy(logits, batch.targets)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for name in collection_feature_dims():
        assert encoder.tokenizer.projections[name].weight.grad.abs().sum() > 0
    assert encoder.tokenizer.type_embeddings.weight.grad.abs().sum() > 0
    assert encoder.encoder.latents.grad.abs().sum() > 0
    assert encoder.encoder.cross_attention.in_proj_weight.grad.abs().sum() > 0
    assert encoder.encoder.layers[0].attention.in_proj_weight.grad.abs().sum() > 0
    before = encoder.encoder.latents.detach().clone()
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    assert not torch.equal(before, encoder.encoder.latents)


def test_padding_and_object_order_do_not_change_embeddings():
    torch.manual_seed(3)
    encoder = FoundationEncoder(config=PerceiverConfig(dropout=0)).eval()
    event = row_to_event_batch(synthetic_row())
    expected = encoder(event)
    altered = copy.deepcopy(event)
    for group in altered.objects.values():
        group.features = torch.cat((group.features.flip(1), torch.full((1, 2, group.features.shape[-1]), float("nan"))), 1)
        group.mask = torch.cat((group.mask.flip(1), torch.zeros((1, 2), dtype=torch.bool)), 1)
    torch.testing.assert_close(encoder(altered), expected, atol=1e-6, rtol=1e-5)
    batch = collate_event_batches([event, row_to_event_batch(synthetic_row(count=0))])
    torch.testing.assert_close(encoder(batch)[:1], expected, atol=1e-6, rtol=1e-5)
    # Rejecting/ignoring padding must hold in backward too.
    encoder(altered).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)


def test_fully_empty_batches_have_finite_embeddings_and_gradients():
    encoder = FoundationEncoder(config=PerceiverConfig(dropout=0))
    no_groups = EventBatch(metadata={"batch_size": 2})
    empty = row_to_event_batch(synthetic_row(count=0))
    zero_length = collate_event_batches([empty, empty])
    all_masked = row_to_event_batch(synthetic_row())
    for group in all_masked.objects.values():
        group.mask[:] = False
        group.features[:] = float("inf")
    for batch in (no_groups, zero_length, all_masked):
        output = encoder(batch)
        assert torch.isfinite(output).all()
        output.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
        encoder.zero_grad(set_to_none=True)


@pytest.mark.parametrize("kwargs", [{"embedding_dim": 7}, {"num_latents": 0}, {"dropout": 1}, {"num_heads": 0}])
def test_perceiver_config_validation(kwargs):
    with pytest.raises(ValueError):
        PerceiverConfig(**kwargs)


@pytest.mark.parametrize("view", ["FullReco", "L1T"])
def test_parquet_stream_paths_caps_provenance_and_labels(tmp_path, view):
    paths = []
    for name in ("a.parquet", "nested/b.parquet"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [synthetic_row(view, offset=index, wrapped=True) for index in range(3)]
        for row in rows:
            row["unread_payload"] = "not a model feature"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
        paths.append(path.resolve())
    assert expand_parquet_paths([tmp_path, str(tmp_path / "**/*.parquet"), paths[0]]) == paths
    events = list(iter_parquet_events(tmp_path, view, max_events=4, read_batch_size=2))
    assert len(events) == 4
    assert [event.metadata["source_row"] for event in events] == [0, 1, 2, 0]
    assert events[-1].metadata["source_file"] == str(paths[1])
    assert list(iter_parquet_events(tmp_path, view, max_events=0)) == []
    sample, labels = load_process_events({"Z": paths[0], "A": paths[1]}, view, max_events_per_class=2)
    assert labels == {"A": 0, "Z": 1}
    assert [event.targets.item() for event in sample] == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="multiple processes"):
        load_process_events({"A": paths[0], "B": paths[0]}, view)
    with pytest.raises(FileNotFoundError, match="No local Parquet"):
        expand_parquet_paths(tmp_path / "missing*.parquet")
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.table({f"{view}_PUPPIPart_PT_mean": [3.0]}), bad)
    with pytest.raises(ValueError, match="B1 scalar summaries"):
        list(iter_parquet_events(bad, view))


def test_stratified_split_is_disjoint_reproducible_and_requires_three_per_class():
    events = [row_to_event_batch(synthetic_row(), target=label) for label in (0, 1) for _ in range(10)]
    splits = stratified_event_split(events, 17)
    assert splits == stratified_event_split(events, 17)
    assert splits != stratified_event_split(events, 18)
    assert {name: len(indices) for name, indices in splits.items()} == {"train": 12, "validation": 4, "test": 4}
    all_indices = [i for indices in splits.values() for i in indices]
    assert sorted(all_indices) == list(range(20))
    for indices in splits.values():
        assert {events[i].targets.item() for i in indices} == {0, 1}
    with pytest.raises(ValueError, match="at least 3"):
        stratified_event_split(events[:2], 0)


@pytest.mark.parametrize("view", ["FullReco", "L1T"])
def test_training_cli_saves_reloadable_best_checkpoint(tmp_path, view):
    root = Path(__file__).resolve().parents[1]
    sources = {}
    for label, name in enumerate(("signal", "background")):
        path = tmp_path / f"{name}.parquet"
        rows = [synthetic_row(view, count=1 + index % 3, offset=label * 30 + index, wrapped=True)
                for index in range(10)]
        pq.write_table(pa.Table.from_pylist(rows), path)
        sources[name] = path
    output = tmp_path / "training"
    command = [sys.executable, str(root / "scripts/train_foundation_smoke.py"),
               "--view", view, "--model-config", str(root / "configs/models/foundation_tiny.yaml"),
               "--max-events-per-class", "10", "--epochs", "2", "--batch-size", "4",
               "--learning-rate", "0.001", "--seed", "7", "--output-dir", str(output)]
    for name, path in sources.items():
        command.extend(["--process", f"{name}={path}"])
    env = dict(os.environ, PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "Epoch 2/2" in result.stdout
    metrics = json.loads((output / "metrics.json").read_text())
    checkpoint = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert metrics["view"] == view
    assert metrics["split_sizes"] == {"train": 12, "validation": 4, "test": 4}
    assert metrics["diagnostics"]["embedding_shape"] == [4, 64]
    assert metrics["diagnostics"]["logits_shape"] == [4, 2]
    assert metrics["diagnostics"]["all_features_finite"]
    assert metrics["diagnostics"]["tokens_embeddings_logits_finite"]
    assert metrics["finite_gradients"]
    assert metrics["files_read"] == sorted(map(str, sources.values()))
    assert len(metrics["event_sources"]) == 20
    assert all(math.isfinite(epoch["train_loss"]) for epoch in metrics["history"])
    best = min(metrics["history"], key=lambda epoch: epoch["validation"]["loss"])
    assert metrics["best_epoch"] == checkpoint["best_epoch"] == best["epoch"]
    for split in ("train", "validation", "test"):
        assert math.isfinite(metrics[split]["loss"])
        assert 0 <= metrics[split]["accuracy"] <= 1
    cfg = dict(checkpoint["config"]["model"])
    cfg.pop("name")
    model = EventClassificationModel(
        FoundationEncoder(checkpoint["collection_feature_dims"], PerceiverConfig(**cfg)), 2,
    ).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    specs = default_collection_specs(checkpoint["config"]["data"]["collection_limits"])
    events, _ = load_process_events(sources, view, specs, max_events_per_class=10)
    test_batch = collate_event_batches([events[i] for i in metrics["split_indices"]["test"]])
    with torch.no_grad():
        logits = model(test_batch)
    assert torch.nn.functional.cross_entropy(logits, test_batch.targets).item() == pytest.approx(metrics["test"]["loss"], abs=1e-6)
    assert (logits.argmax(1) == test_batch.targets).float().mean().item() == pytest.approx(metrics["test"]["accuracy"])
    config = yaml.safe_load((root / "configs/models/foundation_tiny.yaml").read_text())
    assert checkpoint["config"] == config
