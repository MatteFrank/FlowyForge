from pathlib import Path

from flowyforge.data_plugins.collide_v2.pipeline_checks import collect_pipeline_artifacts, file_exists


def test_file_exists(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("ok", encoding="utf-8")

    assert file_exists(path) is True
    assert file_exists(tmp_path / "missing.txt") is False


def test_collect_pipeline_artifacts(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    (processed_dir / "vectorized").mkdir(parents=True)
    (processed_dir / "preprocessed").mkdir()
    (processed_dir / "training" / "classification_mlp").mkdir(parents=True)
    (processed_dir / "evaluation" / "classification_mlp").mkdir(parents=True)
    (processed_dir / "vectorized" / "X.npy").write_bytes(b"x")
    (processed_dir / "vectorized" / "feature_map.json").write_text("{}\n", encoding="utf-8")
    (processed_dir / "preprocessed" / "X_preprocessed.npy").write_bytes(b"x")
    (processed_dir / "training" / "classification_mlp" / "metrics.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    artifacts = collect_pipeline_artifacts(processed_dir)

    assert artifacts["vectorized_x"] is True
    assert artifacts["vectorized_y"] is False
    assert artifacts["feature_map"] is True
    assert artifacts["preprocessed_x"] is True
    assert artifacts["preprocessed_y"] is False
    assert artifacts["training_checkpoint"] is False
    assert artifacts["training_metrics"] is True
    assert artifacts["evaluation_metrics"] is False

