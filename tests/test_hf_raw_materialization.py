"""Raw HF preparation with a mocked stream; no network or external datasets."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
import yaml

from flowyforge.data_plugins.collide_v2.hf_collide1m import (
    materialize_hf_collide1m_multi_process_subset, materialize_hf_collide1m_subset,
    prepare_hf_source,
)
from flowyforge.data_plugins.collide_v2.streaming import StreamingDataset, build_manifest


def mock_stream(monkeypatch, fail_process=None):
    calls = []

    def load_dataset(name, **kwargs):
        calls.append((name, kwargs))
        assert kwargs["streaming"] is True
        assert "columns" not in kwargs  # Keep both views and every object feature.
        if kwargs["data_dir"] == fail_process:
            raise OSError("synthetic remote failure")
        for index in range(20):
            row = {}
            for view in ("FullReco", "L1T"):
                row[f"{view}_PUPPIMET_MET"] = [] if index == 0 else [float(index)]
                row[f"{view}_PUPPIMET_Phi"] = [] if index == 0 else [0.25]
            if index > 0:
                row["late_optional_column"] = [1.0]
            yield row

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    return calls


def test_raw_export_preserves_nested_fields_and_handles_initial_empty_lists(tmp_path, monkeypatch):
    calls = mock_stream(monkeypatch)
    manifest = materialize_hf_collide1m_subset(
        "example/collide", "train", tmp_path / "raw", data_dir="a", max_rows=12, max_files=2,
        cache_dir=tmp_path / "cache",
    )
    assert manifest["rows_written"] == 12
    assert manifest["materialization_mode"] == "raw"
    assert len(manifest["parquet_files"]) == 2
    assert calls[0][1]["cache_dir"] == str(tmp_path / "cache")
    tables = [pq.read_table(tmp_path / "raw" / filename) for filename in manifest["parquet_files"]]
    assert tables[0].schema == tables[1].schema
    rows = [row for table in tables for row in table.to_pylist()]
    assert rows[0]["FullReco_PUPPIMET_MET"] == []
    assert rows[1]["FullReco_PUPPIMET_MET"] == [1.0]
    assert rows[1]["L1T_PUPPIMET_MET"] == [1.0]
    assert rows[0]["late_optional_column"] is None
    assert rows[1]["late_optional_column"] == [1.0]
    original = (tmp_path / "raw" / manifest["parquet_files"][0]).read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_hf_collide1m_subset("example/collide", "train", tmp_path / "raw", data_dir="a")
    assert (tmp_path / "raw" / manifest["parquet_files"][0]).read_bytes() == original


def test_prepare_cli_export_is_readable_by_the_perceiver_stream(tmp_path, monkeypatch):
    mock_stream(monkeypatch)
    from scripts.prepare_collide_source import main

    config = {"paths": {
        "dataset_backend": "hf", "hf_dataset_name": "example/collide", "hf_split": "train",
        "hf_data_dirs": ["a", "b"], "dataset_dir": str(tmp_path / "raw"),
    }, "data": {"max_rows_per_process": 12, "max_files": 2}}
    path = tmp_path / "prepare.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv", ["prepare_collide_source.py", "--config", str(path)])
    assert main() == 0
    export = json.loads((tmp_path / "raw/hf_multi_subset_manifest.json").read_text())
    assert export["rows_written_per_process"] == {"a": 12, "b": 12}
    assert export["errors"] == {}
    limits = {"PUPPIPart": 0, "JetPuppiAK4": 0, "Electron": 0,
              "MuonTight": 0, "PhotonTight": 0, "PUPPIMET": 1}
    for view in ("FullReco", "L1T"):
        manifest = build_manifest({"root": str(tmp_path / "raw")}, view=view, collection_limits=limits)
        events = list(StreamingDataset(manifest, "test"))
        assert len(events) == manifest["split_counts"]["test"]
        assert {event.targets.item() for event in events} == {0, 1}


def test_partial_export_is_an_error_with_a_failure_manifest(tmp_path, monkeypatch):
    mock_stream(monkeypatch, fail_process="b")
    with pytest.raises(RuntimeError, match="Incomplete HF export"):
        materialize_hf_collide1m_multi_process_subset("example/collide", "train", tmp_path / "raw", ["a", "b"])
    manifest = json.loads((tmp_path / "raw/hf_multi_subset_manifest.json").read_text())
    assert manifest["processes_materialized"] == 1
    assert "synthetic remote failure" in manifest["errors"]["b"]


def test_limits_colliding_names_and_obsolete_config_fail_before_download(tmp_path):
    with pytest.raises(ValueError, match="max_rows"):
        materialize_hf_collide1m_subset("dataset", "train", tmp_path, max_rows=101, data_dir="a")
    with pytest.raises(ValueError, match="distinct"):
        materialize_hf_collide1m_multi_process_subset("dataset", "train", tmp_path, ["a/b", "a_b"])
    with pytest.raises(ValueError, match="raw-only"):
        prepare_hf_source({"paths": {}, "preprocessing": {}})
    with pytest.raises(ValueError, match="Only raw"):
        prepare_hf_source({"paths": {"dataset_backend": "hf"}, "data": {"hf_materialization_mode": "summary"}})
