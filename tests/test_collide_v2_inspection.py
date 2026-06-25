import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from flowyforge.data_plugins.collide_v2.eos_paths import list_parquet_files
from flowyforge.data_plugins.collide_v2.schema_inspector import (
    inspect_dataset_schema,
    inspect_parquet_schema,
    write_schema_report,
)


def _write_parquet(path: Path, data: dict[str, list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)


def test_list_parquet_files(tmp_path: Path) -> None:
    first = tmp_path / "a.parquet"
    nested = tmp_path / "nested" / "b.parquet"
    ignored = tmp_path / "nested" / "notes.txt"
    _write_parquet(first, {"event_id": [1], "pt": [10.0]})
    _write_parquet(nested, {"event_id": [2], "pt": [20.0]})
    ignored.parent.mkdir(parents=True, exist_ok=True)
    ignored.write_text("not parquet", encoding="utf-8")

    parquet_files = list_parquet_files(tmp_path)

    assert parquet_files == sorted([first, nested])
    assert list_parquet_files(tmp_path, max_files=1) == [sorted([first, nested])[0]]


def test_list_parquet_files_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Base path does not exist"):
        list_parquet_files(tmp_path / "missing")


def test_inspect_parquet_schema(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    _write_parquet(parquet_path, {"event_id": [1, 2], "pt": [10.0, 20.0], "label": [0, 1]})

    report = inspect_parquet_schema(parquet_path)

    assert report["path"] == str(parquet_path)
    assert report["columns"] == ["event_id", "pt", "label"]
    assert report["num_columns"] == 3
    assert "event_id" in report["schema"]


def test_inspect_dataset_schema_and_write_report(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    bad = tmp_path / "bad.parquet"
    _write_parquet(first, {"event_id": [1], "pt": [10.0]})
    _write_parquet(second, {"event_id": [2], "eta": [0.2]})
    bad.write_text("not a parquet file", encoding="utf-8")

    report = inspect_dataset_schema([first, second, bad])

    assert report["num_files_requested"] == 3
    assert report["num_files_inspected"] == 2
    assert report["num_files_failed"] == 1
    assert report["union_columns"] == ["eta", "event_id", "pt"]
    assert any("same columns" in warning for warning in report["warnings"])
    assert any("could not be read" in warning for warning in report["warnings"])

    output_json = tmp_path / "outputs" / "schema" / "schema_report.json"
    output_txt = tmp_path / "outputs" / "schema" / "schema_report.txt"
    write_schema_report(report, output_json, output_txt)

    assert output_json.exists()
    assert output_txt.exists()
    saved = json.loads(output_json.read_text(encoding="utf-8"))
    assert saved["union_columns"] == ["eta", "event_id", "pt"]


def test_inspect_dataset_schema_raises_when_no_files_readable(tmp_path: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_text("not a parquet file", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No readable parquet files found"):
        inspect_dataset_schema([bad])


def test_inspect_dataset_script(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    _write_parquet(data_dir / "one.parquet", {"event_id": [1], "pt": [10.0]})
    _write_parquet(data_dir / "two.parquet", {"event_id": [2], "pt": [20.0]})

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "backend": "local",
                    "base_path": str(data_dir),
                    "max_files": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    output_json = tmp_path / "reports" / "schema" / "schema_report.json"
    output_txt = tmp_path / "reports" / "schema" / "schema_report.txt"

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{repo_root / 'src'}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(repo_root / "src")
    )
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "inspect_dataset.py"),
            "--config",
            str(config_path),
            "--output-json",
            str(output_json),
            "--output-txt",
            str(output_txt),
            "--max-files",
            "1",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Backend: local" in result.stdout
    assert "Parquet files found: 2" in result.stdout
    assert "Files inspected: 1" in result.stdout
    assert output_json.exists()
    assert output_txt.exists()
