import json
from pathlib import Path

import pandas as pd

from flowyforge.data_plugins.collide_v2.manifest import (
    create_split_manifest,
    scan_parquet_event_counts,
    write_file_event_counts,
    write_split_manifest,
)


def _write_parquet(path: Path, rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "event_id": list(range(rows)),
            "pt": [float(index) for index in range(rows)],
        }
    ).to_parquet(path, index=False)


def test_manifest_generation_on_tiny_parquet_files(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    _write_parquet(dataset_dir / "proc_a" / "a.parquet", rows=3)
    _write_parquet(dataset_dir / "proc_b" / "b.parquet", rows=4)

    counts = scan_parquet_event_counts(dataset_dir)

    assert counts["total_files"] == 2
    assert counts["total_rows"] == 7
    assert {item["path"] for item in counts["files"]} == {
        "proc_a/a.parquet",
        "proc_b/b.parquet",
    }
    assert {item["process"] for item in counts["files"]} == {"proc_a", "proc_b"}

    counts_path = tmp_path / "manifests" / "file_event_counts.json"
    write_file_event_counts(counts, counts_path)
    assert json.loads(counts_path.read_text(encoding="utf-8"))["total_rows"] == 7

    split_manifest = create_split_manifest(
        counts=counts,
        split_counts=None,
        classnames=None,
        folder_map=None,
        seed=123,
    )
    assert split_manifest["classnames"] == ["proc_a", "proc_b"]
    assert len(split_manifest["splits"]["train"]) == 2
    assert split_manifest["splits"]["val"] == []
    assert split_manifest["splits"]["test"] == []

    split_path = tmp_path / "manifests" / "split_manifest.json"
    write_split_manifest(split_manifest, split_path)
    assert json.loads(split_path.read_text(encoding="utf-8"))["seed"] == 123

