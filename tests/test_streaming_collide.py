"""Tiny synthetic Parquet tests for sampling, partitioning and data contracts."""

import math
import os
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader

from flowyforge.data_plugins.collide_v2 import streaming
from flowyforge.data_plugins.collide_v2.streaming import StreamingDataset, build_manifest, inspect_dataset
from flowyforge.data_plugins.collide_v2.object_dataset import collate_event_batches
from flowyforge.training.classification_run import _init_worker


LIMITS = {"PUPPIPart": 0, "JetPuppiAK4": 0, "Electron": 0,
          "MuonTight": 0, "PhotonTight": 0, "PUPPIMET": 1}


def make_sources(tmp_path, rows=30):
    for label in ("a", "b"):
        directory = tmp_path / label
        directory.mkdir()
        for part in range(3):
            data = [{"FullReco_PUPPIMET_MET": [float(i + part * rows + 1)],
                     "FullReco_PUPPIMET_Phi": [0.1]} for i in range(rows)]
            pq.write_table(pa.Table.from_pylist(data), directory / f"{part}.parquet", row_group_size=7)
    return {"root": str(tmp_path)}


def identities(events):
    return {(event.metadata["source_file"], event.metadata["source_row"]) for event in events}


def test_permutation_is_exact_for_non_power_of_two_domains():
    for size in (3, 5, 7, 16, 19, 65, 127):
        ranks = streaming._permuted_ranks(np.arange(size), size, 17)
        assert sorted(ranks.tolist()) == list(range(size))
        assert np.array_equal(ranks, streaming._permuted_ranks(np.arange(size), size, 17))


def test_uniform_cap_all_files_replay_disjoint_splits_and_epoch_order(tmp_path):
    source = {**make_sources(tmp_path), "max_events_per_class": 24}
    manifest = build_manifest(source, collection_limits=LIMITS, seed=19)
    assert manifest == build_manifest(source, collection_limits=LIMITS, seed=19)
    for process in manifest["processes"]:
        selected = process["selected_indices"]
        assert len(selected) == len(set(selected)) == 24
        assert any(index >= 60 for index in selected)
    splits = {name: list(StreamingDataset(manifest, name, read_batch_size=5, shuffle_buffer=4))
              for name in ("train", "validation", "test")}
    sets = {name: identities(events) for name, events in splits.items()}
    assert all(len(splits[name]) == manifest["split_counts"][name] for name in sets)
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["test"] & sets["validation"]
    assert len(set.union(*sets.values())) == 48
    train = StreamingDataset(manifest, "train", read_batch_size=5, shuffle_buffer=4)
    train.set_epoch(2)
    later = list(train)
    assert identities(later) == sets["train"]
    assert [event.metadata for event in later] != [event.metadata for event in splits["train"]]


def test_full_mode_has_no_event_index_and_workers_do_not_duplicate(tmp_path, monkeypatch):
    manifest = build_manifest(make_sources(tmp_path, rows=8), collection_limits=LIMITS)
    assert all(process["selected_indices"] is None for process in manifest["processes"])
    expected = identities(StreamingDataset(manifest, "train", read_batch_size=3))
    parts = []
    for worker_id in range(3):
        monkeypatch.setattr(streaming, "get_worker_info", lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=3))
        parts.append(identities(StreamingDataset(manifest, "train", read_batch_size=3)))
    assert set.union(*parts) == expected
    assert sum(len(part) for part in parts) == len(expected)


def test_spawn_workers_replay_the_same_rows_as_single_process(tmp_path):
    manifest = build_manifest(make_sources(tmp_path, rows=3), collection_limits=LIMITS)
    expected = identities(StreamingDataset(manifest, "train", read_batch_size=2))
    loader = DataLoader(StreamingDataset(manifest, "train", read_batch_size=2),
                        batch_size=2, num_workers=2, multiprocessing_context="spawn",
                        collate_fn=collate_event_batches, worker_init_fn=_init_worker, timeout=15)
    sources = [metadata for batch in loader for metadata in batch.metadata["events"]]
    actual = {(row["source_file"], row["source_row"]) for row in sources}
    assert actual == expected
    assert len(actual) == len(sources)


def test_audit_units_missing_values_and_truncation(tmp_path):
    processes = {}
    for name in ("a", "b"):
        path = tmp_path / f"{name}.parquet"
        row = {"FullReco_PUPPIMET_MET": [1000.0, None], "FullReco_PUPPIMET_Phi": [90.0, 0.0]}
        pq.write_table(pa.Table.from_pylist([row] * 6), path)
        processes[name] = str(path)
    manifest = build_manifest({"processes": processes}, collection_limits=LIMITS,
                              units={"momentum": "MeV", "angle": "deg"})
    event = next(iter(StreamingDataset(manifest, "train")))
    features = event.objects["PUPPIMET"].features[0, 0]
    assert features[0].item() == pytest.approx(math.log1p(1.0))
    assert features[1].item() == pytest.approx(1.0)
    audit = inspect_dataset(manifest, audit_events_per_class=2, read_batch_size=2)
    assert audit["num_events"] == 4
    assert audit["collections"]["PUPPIMET"]["truncated_objects_total"] == 4
    assert audit["collections"]["PUPPIMET"]["nonfinite_values_replaced"] == 4
    assert audit["unit_assumed_fields"] == 4
    assert audit["fields"]["FullReco_PUPPIMET_MET"]["min"] == 1000.0
    assert audit["warnings"]


def test_metadata_units_and_schema_errors_are_checked_before_training(tmp_path):
    source = make_sources(tmp_path, rows=3)
    path = tmp_path / "a/0.parquet"
    schema = pa.schema([pa.field("FullReco_PUPPIMET_MET", pa.list_(pa.float64()), metadata={b"unit": b"MeV"}),
                        pa.field("FullReco_PUPPIMET_Phi", pa.list_(pa.float64()), metadata={b"unit": b"rad"})])
    pq.write_table(pa.Table.from_pylist([{"FullReco_PUPPIMET_MET": [1.0], "FullReco_PUPPIMET_Phi": [0.0]}] * 3, schema=schema), path)
    with pytest.raises(ValueError, match="unit metadata"):
        build_manifest(source, collection_limits=LIMITS)
    pq.write_table(pa.table({"FullReco_PUPPIMET_MET": [1.0], "FullReco_PUPPIMET_Phi": [0.0]}), path)
    with pytest.raises(ValueError, match="numeric object list"):
        build_manifest(source, collection_limits=LIMITS)


def test_file_overlap_fingerprint_and_reader_changes_are_rejected(tmp_path):
    source = make_sources(tmp_path, rows=3)
    path = str(tmp_path / "a/0.parquet")
    with pytest.raises(ValueError, match="multiple processes"):
        build_manifest({"processes": {"a": path, "b": path}}, collection_limits=LIMITS)
    manifest = build_manifest(source, collection_limits=LIMITS)
    with pytest.raises(ValueError, match="frozen"):
        StreamingDataset(manifest, "test", view="L1T")
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match="changed"):
        list(StreamingDataset(manifest, "train"))


def test_known_production_wrapper_and_ambiguous_layout(tmp_path):
    production = tmp_path / "production_final"
    production.mkdir()
    make_sources(production, rows=3)
    manifest = build_manifest({"root": str(tmp_path)}, collection_limits=LIMITS)
    assert manifest["label_map"] == {"a": 0, "b": 1}
    (tmp_path / "production_other").mkdir()
    with pytest.raises(ValueError, match="Ambiguous"):
        build_manifest({"root": str(tmp_path)}, collection_limits=LIMITS)
