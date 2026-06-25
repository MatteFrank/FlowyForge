from pathlib import Path

from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.source_resolver import ResolvedDatasetSource


def test_pploner_adapter_creates_expected_manifest_paths(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    processed_dir = tmp_path / "processed"
    tmp_dir = tmp_path / "tmp"
    dataset_dir.mkdir()

    source = ResolvedDatasetSource(
        backend="local",
        dataset_dir=dataset_dir,
        processed_data_dir=processed_dir,
        tmp_data_dir=tmp_dir,
        parquet_files=[],
        process_folders=[],
    )

    paths = prepare_pploner_paths(source)

    assert paths.dataset_dir == dataset_dir
    assert paths.tmp_data_dir == tmp_dir
    assert paths.processed_data_dir == processed_dir
    assert paths.file_event_counts_path == processed_dir / "manifests" / "file_event_counts.json"
    assert paths.split_manifest_path == processed_dir / "manifests" / "split_manifest.json"
    assert tmp_dir.exists()
    assert (processed_dir / "manifests").exists()

