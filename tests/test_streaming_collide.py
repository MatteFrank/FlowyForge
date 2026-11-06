"""Tiny synthetic Parquet tests for sampling, partitioning and data contracts."""

import math
import os
import copy
import shutil
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader

from flowyforge.data_plugins.collide_v2 import streaming
from flowyforge.data_plugins.collide_v2.streaming import StreamingDataset, build_manifest, inspect_dataset
from flowyforge.data_plugins.collide_v2.provenance import with_origin, sha256_file
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
                     "FullReco_PUPPIMET_Phi": [0.1], "L1T_PUPPIMET_MET": [float(i + 2)],
                     "L1T_PUPPIMET_Phi": [0.2]} for i in range(rows)]
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


def partition_ids(binding, split, **kwargs):
    return {event.metadata["event_id"] for event in StreamingDataset(binding, split, **kwargs)}


@pytest.mark.parametrize("cap", [None, 11])
def test_portable_partitions_reproducible_disjoint_shared_views_and_no_resampling(tmp_path, monkeypatch, cap):
    source = {**make_sources(tmp_path, rows=5), "max_events_per_class": cap}
    manifest = streaming.create_partition_manifest(source, seed=11)
    assert manifest == streaming.create_partition_manifest(source, seed=11, read_batch_size=3)
    other = streaming.create_partition_manifest(source, seed=12)
    memberships = lambda m: {e["id"]: e["split"] for p in m["processes"] for e in p["events"]}
    assert memberships(manifest) != memberships(other)
    assert all(p["split_counts"] == ({"train": 9, "validation": 3, "test": 3} if cap is None else
                                      {"train": 7, "validation": 2, "test": 2}) for p in manifest["processes"])
    target = tmp_path / "partitions.json"
    streaming.save_partition_manifest(manifest, target)
    saved = target.read_bytes()
    with pytest.raises(FileExistsError):
        streaming.save_partition_manifest(other, target)
    assert target.read_bytes() == saved
    assert str(tmp_path) not in target.read_text()
    manifest = streaming.load_partition_manifest(target)
    # Reuse must never call the cap sampler or the split permutation.
    def no_sampling(*args, **kwargs):
        raise AssertionError("Stored partitions must not be resampled")
    monkeypatch.setattr(streaming.random.Random, "sample", no_sampling)
    monkeypatch.setattr(streaming, "_permuted_ranks", no_sampling)
    full = streaming.bind_partition_manifest(manifest, source, collection_limits=LIMITS)
    l1t = streaming.bind_partition_manifest(manifest, source, collection_limits=LIMITS, view="L1T", eta_clip=4.)
    sets = {split: partition_ids(full, split) for split in streaming.SPLITS}
    assert sets == {split: partition_ids(l1t, split) for split in streaming.SPLITS}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["test"] & sets["validation"]
    assert sum(map(len, sets.values())) == 2 * (cap or 15)
    assert partition_ids(full, "train", purpose="pretraining") == sets["train"]
    assert partition_ids(full, "validation", purpose="checkpoint_selection") == sets["validation"]
    for split in ("validation", "test"):
        with pytest.raises(ValueError, match="train split only"):
            StreamingDataset(full, split, purpose="pretraining")
    with pytest.raises(ValueError, match="validation split only"):
        StreamingDataset(full, "test", purpose="checkpoint_selection")
    worker_sets = []
    for worker_id in range(3):
        monkeypatch.setattr(streaming, "get_worker_info", lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=3))
        worker_sets.append(partition_ids(full, "train"))
    assert set.union(*worker_sets) == sets["train"]
    assert sum(map(len, worker_sets)) == len(sets["train"])


def test_partition_identity_survives_relocation_reexport_and_row_reordering(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    source = make_sources(raw, rows=4)
    partitions = streaming.create_partition_manifest(source)
    initial = streaming.bind_partition_manifest(partitions, source, collection_limits=LIMITS)
    expected = {split: partition_ids(initial, split) for split in streaming.SPLITS}
    moved = tmp_path / "moved"
    shutil.copytree(raw, moved)
    relocated = streaming.bind_partition_manifest(partitions, {"root": moved}, collection_limits=LIMITS)
    assert expected == {split: partition_ids(relocated, split) for split in streaming.SPLITS}
    exported = tmp_path / "exported"
    for name in ("a", "b"):
        tables = []
        for path in sorted((raw / name).glob("*.parquet")):
            table = pq.read_table(path)
            tables.append(with_origin(table, {"source": "local", "revision": None, "process": name,
                "file": path.as_uri(), "sha256": sha256_file(path)}, list(range(len(table)))))
        table = pa.concat_tables(tables)
        table = table.take(pa.array(list(reversed(range(len(table))))))
        (exported / name).mkdir(parents=True)
        pq.write_table(table, exported / name / "reencoded.parquet", compression="gzip", row_group_size=2)
    rebound = streaming.bind_partition_manifest(partitions, {"root": exported}, collection_limits=LIMITS)
    assert expected == {split: partition_ids(rebound, split) for split in streaming.SPLITS}
    for event in StreamingDataset(rebound, "train"):
        assert event.metadata["origin"]["file"].startswith("file:")
    # A retained identity with changed features must fail too.
    path = exported / "a/reencoded.parquet"
    table = pq.read_table(path)
    table = table.set_column(table.schema.get_field_index("L1T_PUPPIMET_MET"), "L1T_PUPPIMET_MET",
                             pa.array([[999.0]] * len(table)))
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="content/schema changed"):
        streaming.bind_partition_manifest(partitions, {"root": exported}, collection_limits=LIMITS)


def test_partitions_reject_incompatible_manifests_caps_sources_and_duplicates(tmp_path):
    source = make_sources(tmp_path, rows=3)
    partitions = streaming.create_partition_manifest(source)
    bad = copy.deepcopy(partitions)
    bad["version"] = 99
    with pytest.raises(ValueError, match="version"):
        streaming.bind_partition_manifest(bad, source, collection_limits=LIMITS)
    bad = copy.deepcopy(partitions)
    bad["processes"][0]["events"][0]["split"] = "test"
    bad["seed"] += 1
    with pytest.raises(ValueError, match="checksum"):
        streaming.validate_partition_manifest(bad)
    bad["sha256"] = streaming._json_digest({k: v for k, v in bad.items() if k != "sha256"})
    bad["processes"][0]["events"][0]["id"] = "invalid"
    bad["sha256"] = streaming._json_digest({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="identity"):
        streaming.validate_partition_manifest(bad)
    with pytest.raises(ValueError, match="cap differs"):
        streaming.bind_partition_manifest(partitions, {**source, "max_events_per_class": 4}, collection_limits=LIMITS)
    path = tmp_path / "a/0.parquet"
    duplicate = tmp_path / "a/duplicate.parquet"
    shutil.copyfile(path, duplicate)
    with pytest.raises(ValueError, match="Duplicate event identity"):
        streaming.bind_partition_manifest(partitions, source, collection_limits=LIMITS)
    duplicate.unlink()
    path.unlink()
    with pytest.raises(ValueError, match="identities changed"):
        streaming.bind_partition_manifest(partitions, source, collection_limits=LIMITS)


def test_partition_source_hash_checked_even_when_stat_is_unchanged(tmp_path):
    source = make_sources(tmp_path, rows=3)
    partitions = streaming.create_partition_manifest(source)
    binding = streaming.bind_partition_manifest(partitions, source, collection_limits=LIMITS)
    path = tmp_path / "a/0.parquet"
    stat = path.stat()
    content = bytearray(path.read_bytes())
    content[10] ^= 1
    path.write_bytes(content)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="content changed"):
        list(StreamingDataset(binding, "train"))


def test_partition_too_small_and_legacy_pretraining_are_rejected(tmp_path):
    source = make_sources(tmp_path, rows=1)
    legacy = build_manifest(source, collection_limits=LIMITS)
    with pytest.raises(ValueError, match="shared partitions"):
        StreamingDataset(legacy, "train", purpose="pretraining")
    for name in ("a", "b"):
        (tmp_path / name / "2.parquet").unlink()
    with pytest.raises(ValueError, match="at least 3"):
        streaming.create_partition_manifest(source)
