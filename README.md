# FlowyForge / M-Fond

FlowyForge is a modular foundation platform for complex scientific data. The first fast-track path targets COLLIDE-2V / HEP workflows while keeping the package extensible to future datasets, task plugins, model plugins, and analysis tools.

Current status: early skeleton plus COLLIDE parquet-source inspection, manifest preparation, minimal tiny-sample vectorization, preprocessing, tiny MLP classification training, and basic evaluation/reporting. It does not access EOS, download full datasets, or implement production ML training yet.

## Install

```bash
pip install -e .
```

## Test

```bash
pytest tests/
```

## Data Pipeline Status

Current fast-track pipeline:

```text
Parquet -> inspect/manifest -> vectorize -> preprocess -> train tiny baseline -> evaluate/report
```

Current supported backends:

- local tiny Parquet sample
- Hugging Face materialized subset
- EOS path on LXPLUS, not yet full-scale vectorization

Docs:

- [Pipeline explained](docs/pipeline_explained.md)
- [Vectorization](docs/vectorization.md)
- [Preprocessing](docs/preprocessing.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [HF smoke pipeline](docs/hf_smoke_pipeline.md)
- [HF multi-process classification](docs/hf_multi_process_classification.md)

## Smoke Commands

```bash
python scripts/make_tiny_parquet_sample.py --config configs/paths/local.yaml
python scripts/prepare_collide_source.py --config configs/paths/local.yaml
python scripts/inspect_dataset.py --config configs/paths/local.yaml
python scripts/vectorize_collide.py --config configs/paths/local.yaml
python scripts/preprocess_collide.py --config configs/paths/local.yaml
python scripts/train_task.py --config configs/paths/local.yaml --task classification --model mlp
python scripts/evaluate_task.py --config configs/paths/local.yaml --task classification --model mlp
```

## HF Small Sample

```bash
pip install datasets huggingface_hub
jupyter notebook notebooks/00_hf_collide1m_quickstart.ipynb
python scripts/prepare_collide_source.py --config configs/paths/hf_collide1m.yaml
python scripts/inspect_dataset.py --config configs/paths/hf_collide1m.yaml
python scripts/vectorize_collide.py --config configs/paths/hf_collide1m.yaml
python scripts/preprocess_collide.py --config configs/paths/hf_collide1m.yaml
```

## HF Smoke Pipeline

The default HF config materializes two tiny process folders, enough to exercise the classification path. Use the listing helper if you want to replace or extend `paths.hf_data_dirs`.

```bash
pip install datasets huggingface_hub
python scripts/list_hf_collide_dirs.py --max-dirs 30
```

Keep at least two entries under `paths.hf_data_dirs` for a tiny classification smoke sample.

```bash
python scripts/prepare_collide_source.py --config configs/paths/hf_collide1m.yaml
python scripts/inspect_dataset.py --config configs/paths/hf_collide1m.yaml
python scripts/vectorize_collide.py --config configs/paths/hf_collide1m.yaml
python scripts/preprocess_collide.py --config configs/paths/hf_collide1m.yaml
python scripts/run_hf_smoke_pipeline.py --config configs/paths/hf_collide1m.yaml
```

If `y.npy` exists with at least two classes, training and evaluation can run. If `y.npy` is absent or contains only one class, the HF smoke pipeline still succeeds as X-only or single-class validation.

## EOS Later On LXPLUS

```bash
python scripts/inspect_dataset.py --config configs/paths/eos.yaml
python scripts/vectorize_collide.py --config configs/paths/eos.yaml
```

Real EOS checks are intentionally deferred to LXPLUS. The pploner-style flow is preserved as `Parquet -> vectorized .npy -> preprocessing -> training`.
