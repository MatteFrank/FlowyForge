# COLLIDE Preprocessing

Preprocessing is the next step after vectorization.

For the broader pipeline, see [pipeline_explained.md](pipeline_explained.md). For the previous stage, see [vectorization.md](vectorization.md).

## Role In The Pipeline

```text
Parquet -> vectorized .npy -> preprocessing -> training
```

B1.8 implements minimal standardization for local and Hugging Face materialized tiny samples.

Input:

```text
processed_data_dir/vectorized/X.npy
```

Output:

```text
processed_data_dir/preprocessed/
```

## Command

```bash
python scripts/preprocess_collide.py --config configs/paths/local.yaml
```

For HF materialized samples:

```bash
python scripts/preprocess_collide.py --config configs/paths/hf_collide1m.yaml
```

## Configuration

```yaml
preprocessing:
  method: standardize
  eps: 1.0e-6
  input_subdir: vectorized
  output_subdir: preprocessed
  copy_labels: true
```

Only `standardize` is currently implemented.

## Standardization

The current transformation is:

```text
X_preprocessed = (X - mean) / safe_std
```

Statistics are computed per feature:

- `mean`
- `std`
- `safe_std`

For nearly constant features:

```text
safe_std = 1.0 when std < eps
```

This avoids division by zero and prevents NaN or infinity values.

## Outputs

The preprocessed directory contains:

- `X_preprocessed.npy`
- optional copied `y.npy`
- `preprocessing_stats.json`
- `preprocessing_manifest.json`
- copied `feature_map.json`
- optional copied `label_map.json`

`preprocessing_stats.json` stores the means, standard deviations, safe standard deviations, `eps`, number of rows, and number of features.

`preprocessing_manifest.json` records input and output paths, whether labels are available, the method, row count, feature count, and the current tiny-sample warning.

## Why Standardization?

HEP features often live on different numerical scales. Early MLP or transformer baselines tend to train more stably when input features have comparable scale. Standardization is the simplest reasonable baseline before adding more domain-specific transforms.

## Current Limitations

- The scaler currently uses all available vectorized rows.
- Future training should fit preprocessing statistics only on the train split.
- EOS-scale preprocessing is not implemented yet.
- Future methods may include log transforms, robust scaling, clipping, and feature-specific transforms.

This prepares the dataset for the next step: minimal classification training.

