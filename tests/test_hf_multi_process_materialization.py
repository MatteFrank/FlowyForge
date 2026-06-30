from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from flowyforge.data_plugins.collide_v2.eos_paths import list_parquet_files
from flowyforge.data_plugins.collide_v2.hf_collide1m import (
    materialize_hf_collide1m_multi_process_subset,
)
from flowyforge.data_plugins.collide_v2.vectorization import (
    vectorization_config_from_dict,
    vectorize_parquet_files,
)


def test_multi_process_materialization_writes_process_ids_and_safe_folders(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows_by_dir = {
        "Proc A": [
            {"FullReco_MET_MET": [1.0, 3.0]},
            {"FullReco_MET_MET": [5.0]},
        ],
        "Proc/B": [
            {"FullReco_MET_MET": [10.0]},
            {"FullReco_MET_MET": [20.0, 30.0]},
        ],
        "Proc:C": [
            {"FullReco_MET_MET": [100.0]},
        ],
    }

    def fake_load_dataset(
        dataset_name: str,
        split: str,
        streaming: bool,
        data_dir: str | None = None,
        columns: list[str] | None = None,
    ) -> list[dict]:
        assert dataset_name == "fastmachinelearning/collide-1m"
        assert split == "train"
        assert streaming is True
        assert columns == ["FullReco_MET_MET"]
        return rows_by_dir[data_dir or ""]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    output_dir = tmp_path / "parquet_export"

    manifest = materialize_hf_collide1m_multi_process_subset(
        dataset_name="fastmachinelearning/collide-1m",
        split="train",
        output_dir=output_dir,
        data_dirs=["Proc A", "Proc/B", "Proc:C"],
        max_rows_per_process=2,
        max_files_per_process=1,
        columns=["FullReco_MET_MET"],
    )

    assert manifest["process_to_id"] == {"Proc A": 0, "Proc/B": 1, "Proc:C": 2}
    assert manifest["rows_written_per_process"] == {"Proc A": 2, "Proc/B": 2, "Proc:C": 1}
    assert manifest["processes_materialized"] == 3
    assert manifest["classification_possible"] is True
    assert sorted(manifest["parquet_files"]) == [
        "Proc_A/part-00000.parquet",
        "Proc_B/part-00000.parquet",
        "Proc_C/part-00000.parquet",
    ]
    assert (output_dir / "Proc_A" / "part-00000.parquet").exists()
    assert (output_dir / "Proc_B" / "part-00000.parquet").exists()
    assert (output_dir / "Proc_C" / "part-00000.parquet").exists()

    first_frame = pd.read_parquet(output_dir / "Proc_A" / "part-00000.parquet")
    second_frame = pd.read_parquet(output_dir / "Proc_B" / "part-00000.parquet")
    third_frame = pd.read_parquet(output_dir / "Proc_C" / "part-00000.parquet")
    assert first_frame["process_id"].tolist() == [0, 0]
    assert second_frame["process_id"].tolist() == [1, 1]
    assert third_frame["process_id"].tolist() == [2]
    assert first_frame["FullReco_MET_MET_mean"].tolist() == [2.0, 5.0]

    saved_manifest = json.loads(
        (output_dir / "hf_multi_subset_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["process_to_id"] == manifest["process_to_id"]


def test_hf_vectorization_config_reads_all_materialized_process_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "parquet_export"
    for process_id, process_name in enumerate(["proc_a", "proc_b"]):
        process_dir = dataset_dir / process_name
        process_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "event_id": [0, 1],
                "process_name": [process_name, process_name],
                "process_id": [process_id, process_id],
                "feature_mean": [float(process_id), float(process_id + 1)],
            }
        ).to_parquet(process_dir / "part-00000.parquet", index=False)

    vector_config = vectorization_config_from_dict(
        {
            "paths": {"dataset_backend": "hf"},
            "data": {
                "max_files": 1,
                "max_rows_per_process": 20,
                "max_rows": None,
                "label_column": "process_id",
                "exclude_columns": ["event_id", "process_name", "view"],
                "output_subdir": "vectorized",
            },
        }
    )

    assert vector_config.max_files is None
    result = vectorize_parquet_files(
        list_parquet_files(dataset_dir),
        tmp_path / "processed" / "vectorized",
        vector_config,
    )

    assert result.y_path is not None
    y = np.load(result.y_path)
    assert sorted(np.unique(y).tolist()) == [0, 1]


def test_multi_process_materialization_records_failed_process_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_load_dataset(
        dataset_name: str,
        split: str,
        streaming: bool,
        data_dir: str | None = None,
        columns: list[str] | None = None,
    ) -> list[dict]:
        if data_dir == "bad_process":
            raise RuntimeError("simulated HF folder failure")
        return [{"FullReco_MET_MET": [1.0, 2.0]}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    output_dir = tmp_path / "parquet_export"

    manifest = materialize_hf_collide1m_multi_process_subset(
        dataset_name="fastmachinelearning/collide-1m",
        split="train",
        output_dir=output_dir,
        data_dirs=["bad_process", "good_process"],
        max_rows_per_process=1,
        max_files_per_process=1,
        columns=["FullReco_MET_MET"],
    )

    assert manifest["processes_materialized"] == 1
    assert manifest["classification_possible"] is False
    assert manifest["rows_written_per_process"] == {"bad_process": 0, "good_process": 1}
    assert "simulated HF folder failure" in manifest["errors"]["bad_process"]
    assert (output_dir / "good_process" / "part-00000.parquet").exists()


def test_multi_process_materialization_raises_when_no_process_succeeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_load_dataset(
        dataset_name: str,
        split: str,
        streaming: bool,
        data_dir: str | None = None,
        columns: list[str] | None = None,
    ) -> list[dict]:
        raise RuntimeError(f"simulated failure for {data_dir}")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    output_dir = tmp_path / "parquet_export"

    with pytest.raises(RuntimeError, match="No HF processes were materialized successfully"):
        materialize_hf_collide1m_multi_process_subset(
            dataset_name="fastmachinelearning/collide-1m",
            split="train",
            output_dir=output_dir,
            data_dirs=["bad_a", "bad_b"],
            max_rows_per_process=1,
            max_files_per_process=1,
            columns=["FullReco_MET_MET"],
        )

    saved_manifest = json.loads(
        (output_dir / "hf_multi_subset_manifest.json").read_text(encoding="utf-8")
    )
    assert saved_manifest["processes_materialized"] == 0
    assert set(saved_manifest["errors"]) == {"bad_a", "bad_b"}
