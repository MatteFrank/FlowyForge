import json
from pathlib import Path

import numpy as np
import pandas as pd

from flowyforge.data_plugins.collide_v2.vectorization import (
    VectorizationConfig,
    build_feature_map,
    load_feature_map,
    vectorization_config_from_dict,
    vectorize_parquet_files,
)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_vectorize_parquet_files_with_labels(tmp_path: Path) -> None:
    first = tmp_path / "proc_a" / "a.parquet"
    second = tmp_path / "proc_b" / "b.parquet"
    _write_parquet(
        first,
        pd.DataFrame(
            {
                "event_id": [1, 2],
                "pt": [10.0, 20.0],
                "eta": [0.1, 0.2],
                "process_id": [1, 1],
                "process_name": ["signal", "signal"],
                "view": ["tiny", "tiny"],
                "nested": [[1], [2]],
            }
        ),
    )
    _write_parquet(
        second,
        pd.DataFrame(
            {
                "event_id": [3],
                "pt": [30.0],
                "eta": [0.3],
                "process_id": [0],
                "process_name": ["background"],
                "view": ["tiny"],
                "nested": [[3]],
            }
        ),
    )
    config = VectorizationConfig(
        feature_columns=None,
        label_column="process_id",
        exclude_columns=["event_id", "process_name", "view"],
        max_files=None,
        max_rows=None,
        output_subdir="vectorized",
    )

    result = vectorize_parquet_files([first, second], tmp_path / "out", config)

    assert result.x_path.exists()
    assert result.y_path is not None
    assert result.y_path.exists()
    assert result.feature_map_path.exists()
    assert result.label_map_path is not None
    assert result.label_map_path.exists()
    assert result.manifest_path.exists()

    x = np.load(result.x_path)
    y = np.load(result.y_path)
    assert x.shape == (3, 2)
    assert x.dtype == np.float32
    assert y.shape == (3,)
    assert y.dtype == np.int64
    assert result.feature_columns == ["pt", "eta"]
    assert load_feature_map(result.feature_map_path) == {"pt": 0, "eta": 1}

    label_map = json.loads(result.label_map_path.read_text(encoding="utf-8"))
    assert label_map == {"0": 0, "1": 1}
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_rows"] == 3
    assert manifest["n_features"] == 2
    assert manifest["has_labels"] is True


def test_vectorize_missing_label_column_writes_x_only(tmp_path: Path) -> None:
    parquet_path = tmp_path / "sample.parquet"
    _write_parquet(
        parquet_path,
        pd.DataFrame(
            {
                "event_id": [1, 2],
                "pt": [10.0, 20.0],
                "eta": [0.1, 0.2],
                "process_name": ["a", "b"],
            }
        ),
    )
    config = VectorizationConfig(
        feature_columns=None,
        label_column="process_id",
        exclude_columns=["event_id", "process_name", "view"],
        max_files=None,
        max_rows=None,
        output_subdir="vectorized",
    )

    result = vectorize_parquet_files([parquet_path], tmp_path / "out", config)

    assert result.y_path is None
    assert result.label_map_path is None
    assert not (result.output_dir / "y.npy").exists()
    assert np.load(result.x_path).shape == (2, 2)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["has_labels"] is False
    assert manifest["label_column"] is None


def test_vectorize_respects_global_max_rows(tmp_path: Path) -> None:
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    _write_parquet(
        first,
        pd.DataFrame({"event_id": [1, 2, 3], "pt": [1.0, 2.0, 3.0], "process_id": [0, 0, 0]}),
    )
    _write_parquet(
        second,
        pd.DataFrame({"event_id": [4, 5, 6], "pt": [4.0, 5.0, 6.0], "process_id": [1, 1, 1]}),
    )
    config = VectorizationConfig(
        feature_columns=None,
        label_column="process_id",
        exclude_columns=["event_id", "process_name", "view"],
        max_files=None,
        max_rows=4,
        output_subdir="vectorized",
    )

    result = vectorize_parquet_files([first, second], tmp_path / "out", config)

    assert result.n_rows == 4
    assert np.load(result.x_path).shape == (4, 1)
    assert np.load(result.y_path).shape == (4,)


def test_vectorization_config_from_dict_defaults() -> None:
    config = vectorization_config_from_dict({"data": {"max_files": 2, "max_rows": 5}})

    assert config.feature_columns is None
    assert config.label_column == "process_id"
    assert config.exclude_columns == ["event_id", "process_name", "view"]
    assert config.max_files == 2
    assert config.max_rows == 5
    assert config.output_subdir == "vectorized"
    assert build_feature_map(["a", "b"]) == {"a": 0, "b": 1}
