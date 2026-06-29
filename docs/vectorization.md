# COLLIDE Vectorization

Vectorization is the stage that converts parquet data into dense numerical `.npy` arrays for early ML baselines.

For the broader pipeline, see [pipeline_explained.md](pipeline_explained.md). The next stage is [preprocessing.md](preprocessing.md).

## Role In The Pipeline

```text
Parquet -> vectorized .npy -> preprocessing -> training
```

B1.7 implements the first conversion step for local and Hugging Face materialized tiny samples. It is intentionally pandas-based and not yet designed for full EOS-scale vectorization.

## Command

```bash
python scripts/vectorize_collide.py --config configs/paths/local.yaml
```

For HF materialized samples:

```bash
python scripts/vectorize_collide.py --config configs/paths/hf_collide1m.yaml
```

## Inputs

The vectorizer consumes:

- `source.parquet_files`
- `paths.processed_data_dir`
- vectorization options in `data`

Relevant config fields:

```yaml
data:
  max_files: 2
  max_rows: 10000
  feature_columns: null
  label_column: process_id
  exclude_columns:
    - event_id
    - process_name
    - view
  output_subdir: vectorized
```

When `feature_columns` is `null`, feature columns are inferred from numeric dataframe columns in stable dataframe order. The label column and excluded metadata columns are removed. Object, string, list, and dict columns are ignored for now.

## Outputs

`scripts/vectorize_collide.py` writes under:

```text
paths.processed_data_dir / data.output_subdir
```

Default output directory:

```text
processed_data_dir/vectorized/
```

Files:

- `X.npy`
- optional `y.npy`
- `feature_map.json`
- optional `label_map.json`
- `vectorization_manifest.json`

## X.npy

`X.npy` is the dense feature matrix.

- rows = events or records
- columns = numeric feature columns
- dtype = `float32`

## y.npy

`y.npy` is written only when the configured label column exists.

The default label column is `process_id`. Labels are encoded as `int64` class IDs using a deterministic mapping sorted by string representation.

If the label column is absent, vectorization still succeeds and writes `X.npy` only.

## feature_map.json

`feature_map.json` maps each feature name to its column index in `X.npy`.

This is useful for debugging, interpretability, and validating preprocessing.

## label_map.json

`label_map.json` maps original label values to integer IDs in `y.npy`.

It is written only when labels are available.

## vectorization_manifest.json

The manifest records:

- input parquet files
- output paths
- row count
- feature count
- feature columns
- label column
- whether labels are present
- a warning that this is minimal local/HF tiny-sample vectorization

## EOS Status

This implementation is not an EOS-scale vectorizer. Full dataset vectorization will likely need chunking, streaming, and LXPLUS or HTCondor orchestration.

For safety, `scripts/vectorize_collide.py` refuses EOS vectorization unless `data.max_files` is explicitly set to a small value (`<=10`).

## Next Step: Preprocessing

After vectorization, run:

```bash
python scripts/preprocess_collide.py --config configs/paths/local.yaml
```

Preprocessing standardizes `X.npy`, copies optional labels and maps, and writes metadata under `processed_data_dir/preprocessed/`.

