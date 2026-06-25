# COLLIDE Data Backends

FlowyForge keeps the COLLIDE fast-track data strategy close to the pploner-style pipeline:

```text
Parquet dataset -> vectorized .npy files -> preprocessing -> training
```

The backend abstraction only decides where parquet files come from and where derived artifacts should be written. Downstream vectorization and training code should consume the same interface:

- `paths.dataset_dir`
- `paths.processed_data_dir`
- `data.datasets_config`
- `data.to_classify`
- `data.process_to_folder`
- `data.train_val_test_split_per_class`

## EOS Backend

`configs/paths/eos.yaml` points at the real CERN EOS dataset:

```bash
python scripts/inspect_dataset.py --config configs/paths/eos.yaml
python scripts/vectorize_collide.py --config configs/paths/eos.yaml
```

EOS paths are treated as normal filesystem paths. There is no EOS-specific Python API in the wrapper, and Codex tasks should not access EOS.

## Local Backend

`configs/paths/local.yaml` is for copied or generated tiny parquet samples:

```bash
python scripts/make_tiny_parquet_sample.py --config configs/paths/local.yaml
python scripts/prepare_collide_source.py --config configs/paths/local.yaml
python scripts/inspect_dataset.py --config configs/paths/local.yaml
python scripts/vectorize_collide.py --config configs/paths/local.yaml
```

This backend is the default smoke-test route on a laptop.

## Hugging Face Backend

`configs/paths/hf_collide1m.yaml` targets `fastmachinelearning/collide-1m`. It is safe by default: the helper uses Hugging Face streaming and materializes only a capped local subset controlled by `data.max_rows` and `data.max_files`.

```bash
pip install datasets huggingface_hub
python scripts/prepare_collide_source.py --config configs/paths/hf_collide1m.yaml
python scripts/inspect_dataset.py --config configs/paths/hf_collide1m.yaml
```

The HF backend must not download the full dataset unless a future explicit config and workflow are added for that purpose.

## Manifests

`prepare_collide_source.py` writes:

- `processed_data_dir/manifests/file_event_counts.json`
- `processed_data_dir/manifests/split_manifest.json`

Row counts are read from parquet metadata where possible, so manifest generation does not need to load full files into memory.

