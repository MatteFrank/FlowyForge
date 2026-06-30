# HF Multi-Process Classification Smoke Test

Single-process HF materialization is useful for backend smoke testing, but it is not a valid classification sample. With one `hf_data_dir`, every materialized row has the same `process_id`, so training should be skipped or fail gracefully.

B1.12 adds safe multi-process materialization for tiny Hugging Face classification smoke tests. The default HF config includes two process folders, and each configured folder is streamed separately, converted to flat scalar summary features, and written under its own local parquet folder:

```text
data/hf_cache/collide-1m/parquet_export/
  WJetsToLNu_13TeV-madgraphMLM-pythia8/
    part-00000.parquet
  AnotherProcess/
    part-00000.parquet
```

The materializer assigns deterministic labels from config order:

```text
first hf_data_dir  -> process_id = 0
second hf_data_dir -> process_id = 1
third hf_data_dir  -> process_id = 2
```

This creates a tiny supervised sample compatible with the existing pipeline:

```text
HF streaming summary parquet -> inspect/manifest -> vectorize -> preprocess -> train/evaluate
```

## Discover Process Folders

Use the listing helper to inspect Hugging Face repository file names without downloading data:

```bash
python scripts/list_hf_collide_dirs.py --max-dirs 30
```

The script calls Hugging Face Hub metadata APIs and prints top-level folders that contain parquet files. If `huggingface_hub` is missing, install only the lightweight HF client:

```bash
pip install huggingface_hub
```

Unauthenticated requests may hit rate limits. Setting `HF_TOKEN` can help, but it is not required for local unit tests.

## Configure Two Or More Processes

`configs/paths/hf_collide1m.yaml` already includes two safe defaults:

```yaml
paths:
  hf_data_dirs:
    - WJetsToLNu_13TeV-madgraphMLM-pythia8
    - DYJetsToLL_13TeV-madgraphMLM-pythia8

data:
  max_files: 1
  max_rows_per_process: 20
  max_rows: null
  hf_materialization_mode: summary
```

Use `scripts/list_hf_collide_dirs.py` if you want to replace or extend this list.
Keep `max_rows_per_process` small. This is a smoke classification sample, not a physics validation dataset.

## Run The Tiny Pipeline

```bash
python scripts/prepare_collide_source.py --config configs/paths/hf_collide1m.yaml
python scripts/inspect_dataset.py --config configs/paths/hf_collide1m.yaml
python scripts/vectorize_collide.py --config configs/paths/hf_collide1m.yaml
python scripts/preprocess_collide.py --config configs/paths/hf_collide1m.yaml
python scripts/run_hf_smoke_pipeline.py --config configs/paths/hf_collide1m.yaml
```

`classification_ready` is true only when `preprocessed/y.npy` exists and contains at least two unique classes. If only one class is present, the smoke pipeline reports `single_class_complete` and skips training.

## Limitations

- Still uses tiny local materialized parquet files.
- Still uses summary scalar features, not full nested object tokenization.
- Not EOS-scale.
- Not a physics-quality benchmark.
- The goal is to validate pipeline wiring before larger training work.
