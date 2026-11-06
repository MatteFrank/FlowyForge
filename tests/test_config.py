from pathlib import Path

from flowyforge.core.config import load_config, load_yaml


def test_load_run_config_yaml() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    config = load_config(repo_root / "configs" / "runs" / "classification.yaml")

    assert config["mode"] == "small"
    assert config["sources"]["small"]["max_events_per_class"] == 128


def test_load_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert load_yaml(config_path) == {}

