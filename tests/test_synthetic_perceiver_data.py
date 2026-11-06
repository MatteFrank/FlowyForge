import os
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from flowyforge.core.config import load_yaml
from flowyforge.data_plugins.collide_v2.streaming import build_manifest, inspect_dataset, StreamingDataset
from flowyforge.data_plugins.collide_v2.synthetic import make_tiny_parquet_sample

REPO = Path(__file__).resolve().parents[1]


def test_synthetic_cli_generates_both_views_and_all_collections(tmp_path):
    output = tmp_path / "nested"
    result = subprocess.run([
        sys.executable, str(REPO / "scripts/make_tiny_parquet_sample.py"),
        "--output-dir", str(output), "--events-per-class", "6", "--files-per-class", "2",
    ], cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO / "src")),
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert len(list(output.rglob("*.parquet"))) == 4
    limits = load_yaml(REPO / "configs/models/foundation_tiny.yaml")["data"]["collection_limits"]
    for view in ("FullReco", "L1T"):
        manifest = build_manifest({"root": str(output)}, view=view, collection_limits=limits)
        assert sum(manifest["split_counts"].values()) == 12
        audit = inspect_dataset(manifest, audit_events_per_class=3)
        assert audit["unit_assumed_fields"] == 0
        event = next(iter(StreamingDataset(manifest, "train")))
        assert set(event.objects) == set(limits)
    with pytest.raises(FileExistsError, match="overwrite"):
        make_tiny_parquet_sample(output)


def test_synthetic_seed_reproduces_nested_values(tmp_path):
    first = make_tiny_parquet_sample(tmp_path / "a", events_per_class=3, files_per_class=1, seed=17)
    second = make_tiny_parquet_sample(tmp_path / "b", events_per_class=3, files_per_class=1, seed=17)
    assert pq.read_table(first[0]).equals(pq.read_table(second[0]))
