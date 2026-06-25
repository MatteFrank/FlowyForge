from pathlib import Path

from flowyforge.core.config import load_config, load_yaml


def test_load_local_mac_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    config = load_config(repo_root / "configs" / "local_mac.yaml")

    assert config["data"]["backend"] == "local"
    assert config["data"]["max_events"] == 500


def test_load_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_yaml(config_path) == {}

