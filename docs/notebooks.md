# Notebooks

Notebooks in this repository are exploratory aids, not the reproducible production interface.

Use them for quick HF or local inspection, schema exploration, and small sanity checks. Keep outputs cleared before committing notebooks, and do not use notebooks to store data samples, generated plots, or long logs.

The reproducible interface remains the scripts:

```bash
python scripts/prepare_collide_source.py --config configs/paths/local.yaml
python scripts/inspect_dataset.py --config configs/paths/local.yaml
python scripts/vectorize_collide.py --config configs/paths/local.yaml
```

For the HF quickstart notebook:

```bash
jupyter notebook notebooks/00_hf_collide1m_quickstart.ipynb
```

The notebook is intended only for HF/local quick inspection and small streaming materialization. Full EOS validation happens later on LXPLUS through scripts and config files.

