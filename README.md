# FlowyForge / M-Fond

FlowyForge is a modular foundation platform for complex scientific data. The first fast-track path targets COLLIDE-2V / HEP workflows while keeping the package extensible to future datasets, task plugins, model plugins, and analysis tools.

Current status: early skeleton plus COLLIDE parquet-source inspection and manifest preparation. It does not access EOS, download full datasets, or implement production ML training yet.

## Install

```bash
pip install -e .
```

## Test

```bash
pytest tests/
```

## Smoke Commands

```bash
python scripts/make_tiny_parquet_sample.py --config configs/paths/local.yaml
python scripts/prepare_collide_source.py --config configs/paths/local.yaml
python scripts/inspect_dataset.py --config configs/paths/local.yaml
python scripts/vectorize_collide.py --config configs/paths/local.yaml
```

## HF Small Sample

```bash
pip install datasets huggingface_hub
jupyter notebook notebooks/00_hf_collide1m_quickstart.ipynb
python scripts/prepare_collide_source.py --config configs/paths/hf_collide1m.yaml
python scripts/inspect_dataset.py --config configs/paths/hf_collide1m.yaml
```

## EOS Later On LXPLUS

```bash
python scripts/inspect_dataset.py --config configs/paths/eos.yaml
python scripts/vectorize_collide.py --config configs/paths/eos.yaml
```

Real EOS checks are intentionally deferred to LXPLUS. The pploner-style flow is preserved as `Parquet -> vectorized .npy -> preprocessing -> training`.
