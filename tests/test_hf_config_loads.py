from pathlib import Path

from flowyforge.core.config import load_yaml
from flowyforge.training.classification_run import load_run_config

REPO = Path(__file__).resolve().parents[1]


def test_local_run_points_to_nested_sample():
    config = load_run_config(REPO / 'configs/runs/local.yaml')
    assert config['source']['processes'] == {
        'background': 'data/samples/perceiver_tiny/background',
        'signal': 'data/samples/perceiver_tiny/signal',
    }
    assert config['model']['name'] == 'foundation_tiny'


def test_hf_preparation_and_training_configs_agree_without_downloading():
    preparation = load_yaml(REPO / 'configs/paths/hf_collide1m_raw.yaml')
    run = load_run_config(REPO / 'configs/runs/hf.yaml')
    paths = preparation['paths']
    assert paths['dataset_backend'] == 'hf'
    assert paths['hf_dataset_name'] == 'fastmachinelearning/collide-1m'
    assert set(run['source']['processes'].values()) == {
        str(Path(paths['dataset_dir']) / name) for name in paths['hf_data_dirs']
    }
    assert preparation['data'] == {'max_rows_per_process': 20, 'max_files': 1}
    assert run['source']['max_events_per_class'] == 20
    assert run['model']['name'] == 'foundation_tiny'
