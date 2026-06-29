from pathlib import Path

import pandas as pd

from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"event_id": [1, 2], "pt": [10.0, 20.0]}).to_parquet(path, index=False)


def test_source_resolver_local_fake_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    processed_dir = tmp_path / "processed"
    tmp_dir = tmp_path / "tmp"
    first = dataset_dir / "proc_a" / "a.parquet"
    second = dataset_dir / "proc_b" / "b.parquet"
    _write_parquet(first)
    _write_parquet(second)

    source = resolve_dataset_source(
        {
            "paths": {
                "dataset_backend": "local",
                "dataset_dir": str(dataset_dir),
                "processed_data_dir": str(processed_dir),
                "tmp_data_dir": str(tmp_dir),
            }
        }
    )

    assert source.backend == "local"
    assert source.dataset_dir == dataset_dir
    assert source.processed_data_dir == processed_dir
    assert source.tmp_data_dir == tmp_dir
    assert source.parquet_files == [first, second]
    assert source.process_folders == [dataset_dir / "proc_a", dataset_dir / "proc_b"]


def test_source_resolver_hf_allows_missing_materialized_dir(tmp_path: Path) -> None:
    source = resolve_dataset_source(
        {
            "paths": {
                "dataset_backend": "hf",
                "hf_dataset_name": "fastmachinelearning/collide-1m",
                "hf_split": "train",
                "hf_data_dir": "WJetsToLNu_13TeV-madgraphMLM-pythia8",
                "hf_data_files": "sample.parquet",
                "local_cache_dir": str(tmp_path / "cache"),
                "dataset_dir": str(tmp_path / "cache" / "parquet_export"),
                "processed_data_dir": str(tmp_path / "processed"),
                "tmp_data_dir": str(tmp_path / "tmp"),
            }
        }
    )

    assert source.backend == "hf"
    assert source.parquet_files == []
    assert source.hf_dataset_name == "fastmachinelearning/collide-1m"
    assert source.hf_split == "train"
    assert source.hf_data_dir == "WJetsToLNu_13TeV-madgraphMLM-pythia8"
    assert source.hf_data_files == "sample.parquet"
    assert source.local_cache_dir == tmp_path / "cache"
