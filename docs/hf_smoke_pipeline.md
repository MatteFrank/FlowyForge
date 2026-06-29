# Hugging Face Smoke Pipeline

B1.11 validates the Hugging Face backend with a very small materialized sample from `fastmachinelearning/collide-1m`.

The HF backend is for smoke validation, not full dataset download. `configs/paths/hf_collide1m.yaml` is intentionally conservative:

- `paths.hf_data_dir` selects one process folder instead of resolving every shard in the dataset.
- `data.max_rows` defaults to `20`, enough to validate the code path without keeping large event objects in memory.
- `data.max_files` defaults to `1`.
- `data.hf_materialization_mode` defaults to `summary`.

The helper always uses `streaming=True`.

## Command

```bash
python scripts/run_hf_smoke_pipeline.py --config configs/paths/hf_collide1m.yaml
```

This runs the small backend flow:

```text
HF streaming subset -> local parquet -> inspect/manifest -> vectorize -> preprocess
```

If labels exist, it also runs tiny MLP training and evaluation.

## Safe Materialization

COLLIDE-1M rows contain variable-length arrays. Writing many raw rows to a pandas frame can use a lot of memory on a laptop. Summary mode avoids that by converting selected array-like event columns into scalar features before parquet export:

```text
FullReco_MET_MET -> FullReco_MET_MET_len
FullReco_MET_MET -> FullReco_MET_MET_mean
FullReco_MET_MET -> FullReco_MET_MET_std
FullReco_MET_MET -> FullReco_MET_MET_min
FullReco_MET_MET -> FullReco_MET_MET_max
FullReco_MET_MET -> FullReco_MET_MET_sum
```

Missing or empty values become `len=0` and numeric summaries of `0.0`. The materialized parquet files are therefore flat, numeric, and compatible with the existing vectorizer.

Raw mode is intentionally capped and should be treated as a debugging escape hatch. It keeps full row structures and can exhaust memory quickly if row or file limits are raised.

Unauthenticated Hugging Face requests may show warnings or rate-limit messages. That is not a pipeline failure by itself. Setting `HF_TOKEN` can help with rate limits, but it is not required for the local no-network tests.

## Outcomes

There are two successful outcomes:

- `supervised_complete`: `y.npy` exists, so training and evaluation ran.
- `x_only_complete`: no configured label column was found, so the pipeline stopped after preprocessing.

X-only mode is not a failure. It still validates:

- schema handling
- parquet materialization
- vectorization
- preprocessing
- future self-supervised or unlabeled workflows

If supervised training is desired, set `data.label_column` to a column that exists in the materialized HF sample.

## Reports

The smoke script writes:

```text
processed_data_dir/reports/hf_smoke_report.json
processed_data_dir/reports/hf_smoke_report.md
```

The report records backend metadata, dataset paths, parquet count, key artifact availability, status, and the next recommended action.

## Safety

The full HF dataset must not be downloaded by default. Keep `hf_data_dir`, `max_rows`, and `max_files` small for this smoke path.
