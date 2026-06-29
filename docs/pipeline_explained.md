# FlowyForge COLLIDE-2V Data Pipeline

## Goal

The fast-track COLLIDE-2V pipeline follows the pploner-style shape:

```text
Parquet dataset -> vectorized .npy files -> preprocessing -> training
```

FlowyForge adds a backend abstraction in front of that shape:

```text
EOS / local / Hugging Face materialized sample -> same downstream interface
```

The intent is that inspection, vectorization, preprocessing, and later training consume the same resolved paths and files no matter where the parquet data came from.

## Why Not Train Directly From Parquet Yet?

Parquet is excellent for storage, column-level access, schema inspection, and moving tabular scientific data around without committing bulky derived arrays.

Early ML baselines, however, usually want dense numerical tensors or arrays. Vectorized `.npy` files are simpler to load, faster to iterate over, and easier to reason about while the model and task layers are still young.

This also matches the reference pploner-style pipeline: first convert parquet into numerical arrays, then preprocess, then train.

## Step 1: Source Resolution

Backend-specific paths live in:

- `configs/paths/eos.yaml`
- `configs/paths/local.yaml`
- `configs/paths/hf_collide1m.yaml`

The resolver entry point is `resolve_dataset_source` in `flowyforge.data_plugins.collide_v2.source_resolver`.

It returns a common source object containing:

- backend name
- dataset directory
- processed data directory
- temporary data directory
- discovered parquet files
- process folders when available
- Hugging Face metadata for the HF backend

Downstream code should not care whether the data came from EOS, a local tiny sample, or a Hugging Face subset. It should use the resolved source and configured output paths.

## Step 2: Inspection And Manifests

`scripts/inspect_dataset.py` checks available parquet schemas and writes schema reports. This is useful before vectorization because it reveals columns, schema mismatches, and missing data assumptions.

`manifest.py` counts parquet rows using metadata where possible and writes file-level manifests such as `file_event_counts.json`. These manifests help understand dataset structure before creating vectorized arrays.

Inspection and manifests answer basic questions:

- Which parquet files exist?
- Which columns are available?
- Do schemas differ across files?
- How many rows/events are in each file?
- Which process folders are present?

## Step 3: Vectorization

Vectorization reads parquet files and writes dense numerical arrays.

Input:

- parquet files from `source.parquet_files`

Output under `processed_data_dir/vectorized/`:

- `X.npy`
- optional `y.npy`
- `feature_map.json`
- optional `label_map.json`
- `vectorization_manifest.json`

### X.npy

`X.npy` is the feature matrix.

- rows = events or records
- columns = numeric feature columns

For the current tiny-sample implementation, numeric scalar columns are selected and metadata columns such as `event_id`, `process_name`, and `view` are excluded by config.

### y.npy

`y.npy` contains target labels for supervised tasks.

The default label column is `process_id`. This is intended for first classification baselines. If the label column does not exist, vectorization still writes `X.npy` and skips `y.npy`, which keeps self-supervised or unlabeled workflows possible.

### feature_map.json

`feature_map.json` maps feature names to column indices in `X.npy`.

This is needed for:

- interpretability
- debugging
- checking preprocessing behavior
- mapping model inputs back to physical or tabular columns

### label_map.json

`label_map.json` maps original label values to integer class IDs used in `y.npy`.

The current mapping is deterministic and sorted by string representation of labels.

### vectorization_manifest.json

`vectorization_manifest.json` records reproducibility information:

- input files
- output paths
- selected feature columns
- label column
- number of rows
- number of features
- whether labels were written

## Step 4: Preprocessing

Preprocessing reads `X.npy` and writes standardized arrays.

Input:

- `processed_data_dir/vectorized/X.npy`

Output under `processed_data_dir/preprocessed/`:

- `X_preprocessed.npy`
- optional copied `y.npy`
- `preprocessing_stats.json`
- `preprocessing_manifest.json`
- copied `feature_map.json`
- optional copied `label_map.json`

Standardization is used because HEP features can have very different scales. MLP and transformer training is usually more stable when input features have comparable scale. Standardization is the simplest baseline:

```text
X_preprocessed = (X - mean) / safe_std
```

For nearly constant features, `safe_std` is set to `1.0` to avoid division by zero.

Current limitations:

- the scaler may use all tiny-sample rows
- real training should fit the scaler only on the train split
- future methods may include log transforms, robust scaling, and feature-specific physics transforms

## Step 5: Training

B1.9 adds a minimal supervised classification baseline.

Command:

```bash
python scripts/train_task.py --config configs/paths/local.yaml --task classification --model mlp
```

Training consumes:

- `X_preprocessed.npy`
- `y.npy`
- feature and label maps
- preprocessing metadata

The current model is a tiny PyTorch MLP trained with cross-entropy loss. It writes `checkpoint.pt`, `metrics.json`, and `training_manifest.json` under `processed_data_dir/training/classification_mlp/`.

This is a pipeline sanity check, not a physics-validated model. Later, training should move more fully through FlowyForge task and model plugins so classification, trigger, masked modeling, calorimeter generation, and foundation-style encoders share a consistent interface.

## Current Limitations

- Works primarily on local/HF tiny samples.
- Not EOS-scale yet.
- No nested object tokenization yet.
- No graph builder yet.
- No clean train/val/test split for preprocessing statistics yet.
- Training is only a tiny MLP classification sanity check.
