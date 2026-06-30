import json
from pathlib import Path

import pandas as pd

from scripts.run_hf_smoke_pipeline import run_hf_smoke_pipeline


def test_hf_smoke_pipeline_x_only_report(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "hf_cache" / "parquet_export"
    processed_dir = tmp_path / "processed"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "pt": [10.0, 20.0, 30.0],
            "eta": [0.1, 0.2, 0.3],
            "process_name": ["a", "b", "c"],
        }
    ).to_parquet(dataset_dir / "part-00000.parquet", index=False)
    config = {
        "paths": {
            "dataset_backend": "hf",
            "hf_dataset_name": "fastmachinelearning/collide-1m",
            "hf_split": "train",
            "local_cache_dir": str(tmp_path / "hf_cache"),
            "dataset_dir": str(dataset_dir),
            "tmp_data_dir": str(tmp_path / "tmp"),
            "processed_data_dir": str(processed_dir),
        },
        "data": {
            "max_files": 1,
            "max_rows": 10,
            "feature_columns": None,
            "label_column": "process_id",
            "exclude_columns": ["event_id", "process_name", "view"],
            "output_subdir": "vectorized",
            "materialize_local_parquet": False,
        },
        "preprocessing": {
            "method": "standardize",
            "eps": 1e-6,
            "input_subdir": "vectorized",
            "output_subdir": "preprocessed",
            "copy_labels": True,
        },
    }

    report = run_hf_smoke_pipeline(config)

    assert report["status"] == "x_only_complete"
    assert report["backend"] == "hf"
    assert report["number_of_parquet_files"] == 1
    assert report["x_npy_exists"] is True
    assert report["x_preprocessed_npy_exists"] is True
    assert report["y_npy_exists"] is False
    assert "process_id" in report["next_recommended_action"]

    report_json = processed_dir / "reports" / "hf_smoke_report.json"
    report_md = processed_dir / "reports" / "hf_smoke_report.md"
    assert report_json.exists()
    assert report_md.exists()
    saved_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved_report["status"] == "x_only_complete"
    assert (processed_dir / "vectorized" / "X.npy").exists()
    assert not (processed_dir / "vectorized" / "y.npy").exists()
    assert (processed_dir / "preprocessed" / "X_preprocessed.npy").exists()

    vector_manifest = json.loads(
        (processed_dir / "vectorized" / "vectorization_manifest.json").read_text(encoding="utf-8")
    )
    assert vector_manifest["has_labels"] is False
    assert "Configured label_column was not found" in vector_manifest["warning"]
    assert "process_id" not in vector_manifest["available_columns"]


def test_hf_smoke_pipeline_single_class_report(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "hf_cache" / "parquet_export"
    processed_dir = tmp_path / "processed"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "pt": [10.0, 20.0, 30.0],
            "process_id": [0, 0, 0],
            "process_name": ["single", "single", "single"],
        }
    ).to_parquet(dataset_dir / "part-00000.parquet", index=False)
    config = {
        "paths": {
            "dataset_backend": "hf",
            "hf_dataset_name": "fastmachinelearning/collide-1m",
            "hf_split": "train",
            "hf_data_dirs": ["single"],
            "local_cache_dir": str(tmp_path / "hf_cache"),
            "dataset_dir": str(dataset_dir),
            "tmp_data_dir": str(tmp_path / "tmp"),
            "processed_data_dir": str(processed_dir),
        },
        "data": {
            "max_files": 1,
            "max_rows_per_process": 10,
            "max_rows": None,
            "feature_columns": None,
            "label_column": "process_id",
            "exclude_columns": ["event_id", "process_name", "view"],
            "output_subdir": "vectorized",
            "materialize_local_parquet": False,
        },
        "preprocessing": {
            "method": "standardize",
            "eps": 1e-6,
            "input_subdir": "vectorized",
            "output_subdir": "preprocessed",
            "copy_labels": True,
        },
    }

    report = run_hf_smoke_pipeline(config)

    assert report["status"] == "single_class_complete"
    assert report["y_npy_exists"] is True
    assert report["n_classes"] == 1
    assert "at least two hf_data_dirs" in report["next_recommended_action"]
    assert (processed_dir / "preprocessed" / "y.npy").exists()
    assert not (processed_dir / "training" / "classification_mlp" / "checkpoint.pt").exists()
