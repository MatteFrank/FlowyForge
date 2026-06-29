from pathlib import Path

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source


def test_local_path_config_loads() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    config = load_config(repo_root / "configs" / "paths" / "local.yaml")

    assert config["paths"]["dataset_backend"] == "local"
    assert config["paths"]["dataset_dir"] == "data/samples/collide2v_tiny"
    assert config["data"]["max_files"] == 2


def test_hf_path_config_loads_without_downloading() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    config = load_config(repo_root / "configs" / "paths" / "hf_collide1m.yaml")
    source = resolve_dataset_source(config)

    assert source.backend == "hf"
    assert source.hf_dataset_name == "fastmachinelearning/collide-1m"
    assert source.hf_split == "train"
    assert source.hf_data_dir == "WJetsToLNu_13TeV-madgraphMLM-pythia8"
    assert isinstance(source.parquet_files, list)
    assert config["data"]["materialize_local_parquet"] is True
    assert config["data"]["max_rows"] == 20
    assert config["data"]["max_files"] == 1
    assert config["data"]["hf_materialization_mode"] == "summary"
