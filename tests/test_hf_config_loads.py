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


def test_real_pilot_configs_agree_without_network_or_local_data():
    from flowyforge.data_plugins.collide_v2.sample_preparation import load_sample_config

    run = load_run_config(REPO / 'configs/runs/collide_pilot.yaml')
    hf = load_sample_config(REPO / 'configs/data/collide_pilot_hf.yaml')
    local = load_sample_config(REPO / 'configs/data/collide_pilot_local.yaml')
    names = {'tt0123j_5f_ckm_LO_MLM_semiLeptonic', 'WJetsToLNu_13TeV-madgraphMLM-pythia8'}
    assert set(run['source']['processes']) == set(hf['source']['processes']) == set(local['source']['processes']) == names
    assert hf['output_dir'] == local['output_dir']
    assert hf['selection']['events_per_process'] == local['selection']['events_per_process'] == 5000
    assert run['source']['max_events_per_class'] is None
    for name, path in run['source']['processes'].items():
        assert Path(path).resolve() == Path(hf['output_dir']) / name
    for config in (hf, local):
        for view in ('FullReco', 'L1T'):
            assert config['readers'][view]['collection_limits'] == run['data']['collection_limits']
            assert config['readers'][view]['units'] == run['data']['units']
