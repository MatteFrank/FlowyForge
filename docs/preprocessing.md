# COLLIDE Preprocessing

B1.8 implements minimal pploner-style preprocessing after vectorization.

The preserved fast-track pipeline is:

```text
Parquet -> vectorized .npy -> preprocessing -> training
```

This step reads `processed_data_dir/vectorized/X.npy`, computes per-feature standardization statistics, and writes a preprocessed array for the next phase: minimal classification training.

## Current Method

The only supported method is:

```yaml
preprocessing:
  method: standardize
  eps: 1.0e-6
  input_subdir: vectorized
  output_subdir: preprocessed
  copy_labels: true
```

For each feature, preprocessing computes:

- `mean`
- `std`
- `safe_std`

If a feature has `std < eps`, `safe_std` is set to `1.0`, which prevents NaN or infinity values for constant columns.

## Outputs

The vectorized directory remains unchanged:

```text
processed_data_dir/vectorized/
  X.npy
  y.npy optional
  feature_map.json
  label_map.json optional
  vectorization_manifest.json
```

The preprocessed directory contains:

```text
processed_data_dir/preprocessed/
  X_preprocessed.npy
  y.npy optional copied from vectorized/
  preprocessing_stats.json
  preprocessing_manifest.json
  feature_map.json copied from vectorized/
  label_map.json optional copied from vectorized/
```

## Important Limitation

The current scaler uses all available vectorized rows. That is acceptable for local/HF tiny-sample smoke work, but the production path should fit preprocessing statistics only on the train split.

EOS-scale preprocessing is also deferred. Full CERN-scale processing will likely require chunked arrays and batch orchestration.

## Commands

```bash
python scripts/vectorize_collide.py --config configs/paths/local.yaml
python scripts/preprocess_collide.py --config configs/paths/local.yaml
```

The preprocessed output prepares the dataset for the next step: minimal classification training.

