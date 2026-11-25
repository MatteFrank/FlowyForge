"""Selected-file preparation, preservation and persistent row provenance offline."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from flowyforge.data_plugins.collide_v2.hf_collide1m import download_hf_files, plan_hf_files
from flowyforge.data_plugins.collide_v2.object_dataset import default_collection_specs, iter_parquet_events
from flowyforge.data_plugins.collide_v2.provenance import ORIGIN_COLUMNS, PREFIX, sha256_file
from flowyforge.data_plugins.collide_v2.sample_preparation import (
    inspect_prepared_sample, load_sample_config, plan_sample, prepare_sample,
)
from flowyforge.data_plugins.collide_v2.streaming import (
    StreamingDataset, bind_partition_manifest, build_manifest, create_partition_manifest,
)

REVISION = "a" * 40


def fixture_config(tmp_path, *, rows=15, selected=9):
    processes = {}
    for process in ("tt", "wjets"):
        paths = []
        for part, count in enumerate((rows // 2, rows - rows // 2)):
            path = tmp_path / "originals" / process / f"{part}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            fields, arrays = [], []
            for view in ("FullReco", "L1T"):
                for name in ("MET", "Phi"):
                    # Match actual HF half-float large-list fields; keep empties.
                    field = pa.field(f"{view}_PUPPIMET_{name}",
                                     pa.large_list(pa.field("item", pa.float16(), nullable=False)),
                                     metadata={b"unit": b"GeV" if name == "MET" else b"rad"})
                    fields.append(field)
                    arrays.append(pa.array([[] if i == 0 else [np.float16(i + part * 20)] for i in range(count)], type=field.type))
            fields.extend([pa.field("event_number", pa.int64()),
                           pa.field("other_detector", pa.large_list(pa.list_(pa.int32())))])
            arrays.extend([pa.array(list(range(part * 100, part * 100 + count))),
                           pa.array([[[1, 2], [], [3]]] * count, type=fields[-1].type)])
            schema = pa.schema(fields, metadata={b"production": b"test-v1", b"opaque": b"\x00\xff"})
            with pq.ParquetWriter(path, schema, use_compliant_nested_type=False) as writer:
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema), row_group_size=3)
                writer.add_key_value_metadata({b"footer_only": f"{process}:{part}".encode()})
            paths.append(str(path))
        processes[process] = paths
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump({"data": {"collection_limits": {
        "PUPPIPart": 0, "JetPuppiAK4": 0, "Electron": 0,
        "MuonTight": 0, "PhotonTight": 0, "PUPPIMET": 1}}}), encoding="utf-8")
    config = {"source": {"kind": "local", "processes": processes},
              "model_config": "model.yaml", "output_dir": str(tmp_path / "sample"),
              "partitions": {"num_workers": 1},
              "selection": {"events_per_process": selected, "seed": 21, "read_batch_size": 2}}
    path = tmp_path / "sample.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, config


def test_cli_export_preserves_raw_schema_metadata_both_views_and_origins(tmp_path, monkeypatch):
    config_path, raw = fixture_config(tmp_path)
    from scripts.prepare_collide_source import main
    monkeypatch.setattr(sys, "argv", ["prepare_collide_source.py", "--config", str(config_path), "--stage", "plan"])
    assert main() == 0
    assert not Path(raw["output_dir"]).exists()
    monkeypatch.setattr(sys, "argv", ["prepare_collide_source.py", "--config", str(config_path)])
    assert main() == 0
    output = Path(raw["output_dir"])
    manifest = json.loads((output / "sample_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "not uniform" in manifest["selection"]["scope"]
    assert {p["exported_rows"] for p in manifest["processes"].values()} == {9}
    source_tables = {Path(path).as_uri(): pq.read_table(path) for paths in raw["source"]["processes"].values() for path in paths}
    origins = set()
    for entry in manifest["output_files"]:
        exported = pq.read_table(output / entry["path"])
        original = source_tables[entry["input_file"]]
        indices = exported[PREFIX + "row"].to_pylist()
        values = exported.select(original.schema.names)
        assert values.schema.equals(original.schema, check_metadata=True)
        assert values.equals(original.take(pa.array(indices)))
        source_path = next(path for paths in raw["source"]["processes"].values() for path in paths if Path(path).as_uri() == entry["input_file"])
        expected_metadata = pq.read_metadata(source_path).metadata
        actual_metadata = pq.read_metadata(output / entry["path"]).metadata
        assert all(actual_metadata[key] == value for key, value in expected_metadata.items() if key != b"ARROW:schema")
        assert exported[PREFIX + "file"].to_pylist() == [entry["input_file"]] * len(exported)
        assert exported[PREFIX + "revision"].null_count == len(exported)
        origins.update((entry["input_file"], row) for row in indices)
    assert len(origins) == 18
    for view in ("FullReco", "L1T"):
        cfg = load_sample_config(config_path)
        stream_manifest = build_manifest({"root": str(output)}, **cfg["readers"][view])
        events = [event for split in ("train", "validation", "test") for event in StreamingDataset(stream_manifest, split)]
        assert {(event.metadata["origin"]["file"], event.metadata["origin"]["row"]) for event in events} == origins
        for event in events:
            origin = event.metadata["origin"]
            original_path = next(path for paths in raw["source"]["processes"].values() for path in paths if Path(path).as_uri() == origin["file"])
            specs = default_collection_specs(cfg["readers"][view]["collection_limits"])
            expected = list(iter_parquet_events(original_path, view, specs=specs))[origin["row"]]
            torch.testing.assert_close(event.objects["PUPPIMET"].features, expected.objects["PUPPIMET"].features)
            assert origin["sha256"] == sha256_file(original_path)
        assert manifest["validation"][view]["events_checked"] == 18
        from flowyforge.data_plugins.collide_v2.streaming import inspect_dataset
        full_audit = inspect_dataset(stream_manifest, audit_events_per_class=9, read_batch_size=2)
        merged = json.loads((output / "validation.json").read_text())[view]
        assert {k: v for k, v in merged.items() if k != "scope"} == {k: v for k, v in full_audit.items() if k != "scope"}
    monkeypatch.setattr(sys, "argv", ["prepare_collide_source.py", "--config", str(config_path), "--stage", "inspect"])
    assert main() == 0
    before = (output / "sample_manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_sample(load_sample_config(config_path))
    assert (output / "sample_manifest.json").read_bytes() == before


def test_reexport_keeps_original_row_identity_and_inspection_detects_changes(tmp_path):
    path, raw = fixture_config(tmp_path)
    first = prepare_sample(load_sample_config(path))
    config = load_sample_config(path)
    config["source"]["processes"] = {name: str(Path(config["output_dir"]) / name) for name in first["processes"]}
    config["output_dir"] = str(tmp_path / "second")
    config["selection"]["events_per_process"] = 5
    second = prepare_sample(config)
    original_files = {Path(file).as_uri() for files in raw["source"]["processes"].values() for file in files}
    for entry in second["output_files"]:
        file = tmp_path / "second" / entry["path"]
        table = pq.read_table(file)
        assert set(table[PREFIX + "file"].to_pylist()) <= original_files
        # Ordinary Arrow save/load must carry origins without consulting sidecars.
        relocated = tmp_path / "relocated.parquet"
        pq.write_table(table, relocated)
        assert pq.read_table(relocated).select(ORIGIN_COLUMNS).equals(table.select(ORIGIN_COLUMNS))
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        inspect_prepared_sample(tmp_path / "second")


def test_shared_partitions_survive_preparation_and_reexport_with_nested_half_floats(tmp_path):
    path, _ = fixture_config(tmp_path, selected=15)
    config = load_sample_config(path)
    partitions = create_partition_manifest(config["source"])
    expected = {split: {event["id"] for process in partitions["processes"] for event in process["events"]
                        if event["split"] == split} for split in ("train", "validation", "test")}
    for destination in ("first", "second"):
        config["output_dir"] = str(tmp_path / destination)
        prepare_sample(config)
        source = {"root": config["output_dir"]}
        for view in ("FullReco", "L1T"):
            binding = bind_partition_manifest(partitions, source, **config["readers"][view])
            assert {split: {event.metadata["event_id"] for event in StreamingDataset(binding, split)}
                    for split in expected} == expected
            specs = default_collection_specs(config["readers"][view]["collection_limits"])
            direct = {event.metadata["event_id"] for file in (tmp_path / destination).rglob("*.parquet")
                      for event in iter_parquet_events(file, view, specs=specs)}
            assert direct == set.union(*expected.values())
        config["source"]["processes"] = {name: str(tmp_path / destination / name) for name in ("tt", "wjets")}


def test_more_than_100_rows_is_valid_and_seed_replays_selection(tmp_path):
    path, _ = fixture_config(tmp_path, rows=105, selected=101)
    config = load_sample_config(path)
    first = prepare_sample(config)
    config["output_dir"] = str(tmp_path / "repeat")
    second = prepare_sample(config)
    assert all(p["exported_rows"] == 101 for p in first["processes"].values())
    for a, b in zip(first["output_files"], second["output_files"]):
        assert pq.read_table(tmp_path / "sample" / a["path"]).equals(pq.read_table(tmp_path / "repeat" / b["path"]))


def test_missing_view_partial_origins_and_insufficient_rows_fail(tmp_path):
    path, raw = fixture_config(tmp_path, selected=50)
    config = load_sample_config(path)
    with pytest.raises(ValueError, match="only 15 rows"):
        prepare_sample(config)
    failed = json.loads((tmp_path / "sample/sample_manifest.json").read_text())
    assert failed["status"] == "failed" and failed["output_files"] == []
    file = raw["source"]["processes"]["tt"][0]
    original = pq.read_table(file)
    pq.write_table(original.drop(["L1T_PUPPIMET_MET"]), file)
    with pytest.raises(ValueError, match="L1T_PUPPIMET_MET"):
        plan_sample(config)
    pq.write_table(original.append_column(PREFIX + "row", pa.array(range(len(original)))), file)
    with pytest.raises(ValueError, match="provenance"):
        plan_sample(config)


def test_invalid_existing_origin_is_not_replaced_during_reexport(tmp_path):
    path, _ = fixture_config(tmp_path)
    config = load_sample_config(path)
    first = prepare_sample(config)
    original_output = Path(config["output_dir"])
    file = original_output / first["output_files"][0]["path"]
    table = pq.read_table(file)
    table = table.set_column(table.schema.get_field_index(PREFIX + "row"), PREFIX + "row",
                             pa.array([-1] * len(table), type=pa.int64()))
    pq.write_table(table, file)
    config["source"]["processes"] = {name: str(original_output / name) for name in first["processes"]}
    config["output_dir"] = str(tmp_path / "invalid_reexport")
    with pytest.raises(ValueError, match="nonnegative"):
        prepare_sample(config)
    assert pq.read_table(file)[PREFIX + "row"].to_pylist() == [-1] * len(table)
    assert json.loads((tmp_path / "invalid_reexport/sample_manifest.json").read_text())["status"] == "failed"


def fake_hf(tmp_path):
    config_path, raw = fixture_config(tmp_path)
    sources = {}
    siblings = []
    for process, paths in raw["source"]["processes"].items():
        for path in paths:
            name = f"{process}/{Path(path).name}"
            sources[name] = Path(path)
            siblings.append(SimpleNamespace(rfilename=name, size=Path(path).stat().st_size,
                            lfs=SimpleNamespace(sha256=sha256_file(path))))
    calls = []
    def info(dataset, **kwargs):
        assert kwargs == {"revision": REVISION, "files_metadata": True}
        calls.append(dataset)
        return SimpleNamespace(sha=REVISION, siblings=siblings)
    def download(**kwargs):
        assert kwargs["revision"] == REVISION and kwargs["repo_type"] == "dataset"
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[kwargs["filename"]], target)
        return str(target)
    processes = {process: [name for name in sources if name.startswith(process + "/")] for process in raw["source"]["processes"]}
    return config_path, SimpleNamespace(dataset_info=info), download, processes, calls


def test_hf_revision_explicit_files_budget_and_original_protection(tmp_path):
    _, api, downloader, processes, calls = fake_hf(tmp_path)
    plan = plan_hf_files("example/collide", REVISION, processes, 1000000, api=api)
    assert calls == ["example/collide"]
    with pytest.raises(ValueError, match="exceeding"):
        plan_hf_files("example/collide", REVISION, processes, plan["selected_bytes"] - 1, api=api)
    for revision in ("main", "v1", "abc"):
        with pytest.raises(ValueError, match="pinned"):
            plan_hf_files("example/collide", revision, processes, 1000000, api=api)
    for name in ("tt/*.parquet", "tt/../outside.parquet", "wjets/0.parquet"):
        with pytest.raises(ValueError, match="Invalid"):
            plan_hf_files("example/collide", REVISION, {"tt": [name]}, 1000000, api=api)
    with pytest.raises(ValueError, match="not found"):
        plan_hf_files("example/collide", REVISION, {"tt": ["tt/missing.parquet"]}, 1000000, api=api)
    files = download_hf_files(plan, tmp_path / "cache", downloader=downloader)
    def no_download(**kwargs):
        pytest.fail("Verified original should be reused without a download")
    assert download_hf_files(plan, tmp_path / "cache", downloader=no_download) == files
    file = Path(files[0]["path"])
    file.write_bytes(b"must not overwrite")
    with pytest.raises(FileExistsError, match="overwrite"):
        download_hf_files(plan, tmp_path / "cache", downloader=no_download)
    assert file.read_bytes() == b"must not overwrite"
    def corrupt_download(**kwargs):
        path = Path(downloader(**kwargs))
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
        return str(path)
    with pytest.raises(ValueError, match="SHA256"):
        download_hf_files(plan, tmp_path / "bad_download", downloader=corrupt_download)


def test_hf_selected_file_export_has_pinned_logical_origins(tmp_path, monkeypatch):
    path, api, downloader, processes, _ = fake_hf(tmp_path)
    from flowyforge.data_plugins.collide_v2 import sample_preparation as preparation
    config = load_sample_config(path)
    config["source"] = {"kind": "hf_files", "dataset": "example/collide", "revision": REVISION,
                         "processes": processes, "max_download_bytes": 1000000,
                         "download_dir": str(tmp_path / "cache")}
    monkeypatch.setattr(preparation, "plan_hf_files", lambda *args: plan_hf_files(*args, api=api))
    monkeypatch.setattr(preparation, "download_hf_files", lambda *args: download_hf_files(*args, downloader=downloader))
    manifest = prepare_sample(config)
    assert manifest["source"]["revision"] == REVISION
    for entry in manifest["output_files"]:
        table = pq.read_table(tmp_path / "sample" / entry["path"])
        assert set(table[PREFIX + "revision"].to_pylist()) == {REVISION}
        assert set(table[PREFIX + "source"].to_pylist()) == {"hf://datasets/example/collide"}
        assert set(table[PREFIX + "file"].to_pylist()) == {entry["input_file"]}
    assert all(Path(file["path"]).is_file() for file in manifest["source"]["files"])


def memberships(manifest):
    return {(process["name"], process["label"], e["id"], e["split"])
            for process in manifest["processes"] for e in process["events"]}


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_fast_parallel_preparation_matches_strict_memberships(tmp_path, workers):
    from flowyforge.data_plugins.collide_v2.streaming import load_partition_manifest
    path, _ = fixture_config(tmp_path)
    config = load_sample_config(path)
    reference = prepare_sample(config)
    original = load_partition_manifest(tmp_path / "sample/partitions.json")
    config["output_dir"] = str(tmp_path / "parallel")
    config["partitions"]["num_workers"] = workers
    parallel = prepare_sample(config)
    fast = load_partition_manifest(tmp_path / "parallel/partitions.json")
    strict = create_partition_manifest({"root": config["output_dir"]}, seed=config["selection"]["seed"])
    assert fast["version"] == 2 and fast["check_integrity_SHA256"] == "fast"
    assert strict["version"] == 1
    assert memberships(original) == memberships(fast) == memberships(strict)
    assert fast["sha256"] != strict["sha256"]
    assert all("content_sha256" not in e for p in fast["processes"] for e in p["events"])
    for a, b in zip(reference["output_files"], parallel["output_files"]):
        assert a["path"] == b["path"]
        assert pq.read_table(tmp_path / "sample" / a["path"]).equals(pq.read_table(tmp_path / "parallel" / b["path"]))
    for view in ("FullReco", "L1T"):
        binding = bind_partition_manifest(fast, {"root": config["output_dir"]}, **config["readers"][view])
        seen = set()
        for split in ("train", "validation", "test"):
            ids = {e.metadata["event_id"] for e in StreamingDataset(binding, split)}
            expected = {identity for _, _, identity, part in memberships(fast) if part == split}
            assert ids == expected and not seen.intersection(ids)
            seen.update(ids)


def test_fast_collects_origins_once_and_never_fingerprints_prepared_payloads(tmp_path, monkeypatch):
    from flowyforge.data_plugins.collide_v2 import sample_preparation as prep, streaming
    from flowyforge.data_plugins.collide_v2.provenance import with_origin
    path, _ = fixture_config(tmp_path)
    config = load_sample_config(path)
    calls = []
    real_hash = prep.sha256_file
    def original_hash(path):
        assert "originals" in Path(path).parts
        calls.append(str(path))
        return real_hash(path)
    monkeypatch.setattr(prep, "sha256_file", original_hash)
    def forbidden(*args, **kwargs):
        pytest.fail("fast must not hash full prepared files or scan features again for partitions")
    with monkeypatch.context() as context:
        context.setattr(streaming, "_scan_partition_source", forbidden)
        prepare_sample(config)
    assert len(calls) == len(set(calls)) == 4  # Initial origin-v1 digest only.
    monkeypatch.setattr(streaming, "sha256_file", forbidden)
    fast = streaming.load_partition_manifest(tmp_path / "sample/partitions.json")
    assert all(t["event_fingerprinting_seconds"] == 0 for t in json.loads((tmp_path / "sample/performance.json").read_text())["file_timings"])
    # Moving bytes and preserving metadata is supported without trusting a cache path.
    shutil.copytree(tmp_path / "sample", tmp_path / "moved")
    for view in ("FullReco", "L1T"):
        binding = bind_partition_manifest(fast, {"root": str(tmp_path / "moved")}, **config["readers"][view])
        for _ in range(2):
            assert len(list(StreamingDataset(binding, "train"))) == fast["split_counts"]["train"]
    # An unsupported Python value in a physical column cannot be converted or hashed in fast records.
    table = with_origin(pa.table({"opaque": [[1, 2]]}), {"source": "local", "revision": None,
        "file": "file:///original", "process": "tt", "sha256": "a" * 64}, [0])
    digest = streaming._json_digest
    def no_event_fingerprint(value):
        assert not isinstance(value, list)
        return digest(value)
    monkeypatch.setattr(streaming, "_json_digest", no_event_fingerprint)
    assert len(streaming._event_records(table, "tt", policy="fast")) == 1


def test_existing_prepared_partitions_stage_projects_only_origins(tmp_path, monkeypatch):
    from flowyforge.data_plugins.collide_v2 import sample_preparation as prep, streaming
    from scripts.prepare_collide_source import main
    path, _ = fixture_config(tmp_path)
    config = load_sample_config(path)
    config["partitions"]["enabled"] = False
    prepare_sample(config)
    output = Path(config["output_dir"])
    before = {p: p.read_bytes() for p in output.rglob("*.parquet")}
    manifest_before = (output / "sample_manifest.json").read_bytes()
    def forbidden(*args, **kwargs):
        pytest.fail("partitions on prepared fast sample must not export, hash payloads or audit physical objects")
    monkeypatch.setattr(prep, "sha256_file", forbidden)
    monkeypatch.setattr(streaming, "sha256_file", forbidden)
    monkeypatch.setattr(prep, "_export_file", forbidden)
    monkeypatch.setattr(prep, "_validate_file", forbidden)
    original_batches = pq.ParquetFile.iter_batches
    def projected(self, *args, **kwargs):
        assert set(kwargs["columns"]) == set(ORIGIN_COLUMNS)
        yield from original_batches(self, *args, **kwargs)
    monkeypatch.setattr(pq.ParquetFile, "iter_batches", projected)
    monkeypatch.setattr(sys, "argv", ["prepare_collide_source.py", "--config", str(path), "--stage", "partitions"])
    assert main() == 0
    assert (output / "partitions.json").is_file()
    assert manifest_before == (output / "sample_manifest.json").read_bytes()
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(FileExistsError, match="overwrite"):
        prep.prepare_partitions(config)


def test_fast_metadata_changes_duplicate_ids_and_invalid_policies_rejected(tmp_path):
    import os
    from flowyforge.data_plugins.collide_v2 import sample_preparation as prep, streaming
    path, _ = fixture_config(tmp_path)
    config = load_sample_config(path)
    assert prep._partition_settings() == {"enabled": True, "check_integrity_SHA256": "fast",
        "validation_fraction": .2, "test_fraction": .2, "read_batch_size": 2048, "num_workers": 4}
    for value in ("FAST", True, "none"):
        with pytest.raises(ValueError, match="exactly"):
            prep._partition_settings({"check_integrity_SHA256": value})
    with pytest.raises(ValueError, match="num_workers"):
        prep._partition_settings({"num_workers": 3})
    prepare_sample(config)
    parts = streaming.load_partition_manifest(tmp_path / "sample/partitions.json")
    file = next((tmp_path / "sample").rglob("*.parquet"))
    stat = file.stat()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
    with pytest.raises(ValueError, match="metadata/schema changed"):
        bind_partition_manifest(parts, {"root": config["output_dir"]}, **config["readers"]["FullReco"])
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    shutil.copy2(file, file.with_name("duplicate.parquet"))
    with pytest.raises(ValueError, match="Duplicate"):
        create_partition_manifest({"root": config["output_dir"]}, check_integrity_SHA256="fast")
    modified = json.loads(json.dumps(parts))
    modified["processes"][0]["events"][0]["content_sha256"] = "0" * 64
    modified["sha256"] = streaming._json_digest({k: v for k, v in modified.items() if k != "sha256"})
    with pytest.raises(ValueError, match="must not claim"):
        streaming.validate_partition_manifest(modified)


def test_parallel_failure_joins_workers_and_removes_partial_outputs(tmp_path):
    path, _ = fixture_config(tmp_path)
    config = load_sample_config(path)
    first = prepare_sample(config)
    original_output = Path(config["output_dir"])
    file = original_output / first["output_files"][-1]["path"]
    table = pq.read_table(file)
    table = table.set_column(table.schema.get_field_index(PREFIX + "process"), PREFIX + "process",
                             pa.array(["wrong_process"] * len(table)))
    pq.write_table(table, file)
    config["source"]["processes"] = {name: str(original_output / name) for name in first["processes"]}
    config["output_dir"] = str(tmp_path / "failed")
    config["partitions"]["num_workers"] = 2
    with pytest.raises(ValueError, match="Origin process"):
        prepare_sample(config)
    failed = tmp_path / "failed"
    assert not list(failed.rglob("*.parquet")) and not list(failed.rglob("*.tmp"))
    assert not (failed / "partitions.json").exists()
    assert json.loads((failed / "sample_manifest.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        prepare_sample(config)


def test_raw_fast_identity_uses_original_digest_then_reuses_metadata(tmp_path, monkeypatch):
    from flowyforge.data_plugins.collide_v2 import streaming
    path, _ = fixture_config(tmp_path)
    config = load_sample_config(path)
    strict = create_partition_manifest(config["source"])
    fast = create_partition_manifest(config["source"], check_integrity_SHA256="fast")
    assert memberships(fast) == memberships(strict)
    assert all("origin_sha256" in f for p in fast["processes"] for f in p["files"])
    def forbidden(*args, **kwargs):
        pytest.fail("Original digest must be reused for raw fast binding after initial inventory")
    monkeypatch.setattr(streaming, "sha256_file", forbidden)
    for view in ("FullReco", "L1T"):
        binding = bind_partition_manifest(fast, config["source"], **config["readers"][view])
        assert len(list(StreamingDataset(binding, "train"))) == fast["split_counts"]["train"]
