# COLLIDE Vectorization

B1.7 adds minimal pploner-style vectorization for local and Hugging Face materialized tiny parquet samples.

The preserved pipeline shape is:

```text
Parquet -> vectorized .npy -> preprocessing -> training
```

This step only implements the first arrow for small samples. It uses pandas to read parquet files, concatenates small dataframes in memory, selects numeric scalar feature columns, and writes numpy arrays.

## Outputs

`scripts/vectorize_collide.py` writes these files under `paths.processed_data_dir / data.output_subdir`:

- `X.npy`
- `y.npy` when the configured label column exists
- `feature_map.json`
- `label_map.json` when labels exist
- `vectorization_manifest.json`

The manifest records input files, output paths, row and feature counts, selected feature columns, label status, and a warning that this is minimal vectorization for local/HF tiny samples.

## Labels And Features

By default, configs use:

```yaml
data:
  feature_columns: null
  label_column: process_id
  exclude_columns:
    - event_id
    - process_name
    - view
```

When `feature_columns` is null, the vectorizer infers numeric columns in dataframe order and excludes the label and configured metadata columns. Object, string, list, and dict columns are ignored.

Labels are encoded as `int64` with a deterministic mapping sorted by string representation.

## EOS Status

This implementation is not an EOS-scale vectorizer. Full dataset vectorization will likely need chunking, streaming, and LXPLUS or HTCondor orchestration.

For safety, `scripts/vectorize_collide.py` refuses EOS vectorization unless `data.max_files` is explicitly set to a small value (`<=10`).

## Next Step: Preprocessing

After vectorization, run:

```bash
python scripts/preprocess_collide.py --config configs/paths/local.yaml
```

B1.8 standardizes `X.npy`, copies optional labels and maps, and writes preprocessing metadata under `paths.processed_data_dir / preprocessing.output_subdir`.
- copy label_map.json to output_dir/label_map.json if present
- write preprocessing_stats.json
- write preprocessing_manifest.json

Manifest should include:
- input_dir
- output_dir
- x_input_path
- x_output_path
- y_available
- method
- n_rows
- n_features
- warning:
  "Minimal preprocessing for local/HF tiny samples; train-only statistics and EOS-scale preprocessing come later."

2. Add script:

scripts/preprocess_collide.py

Behavior:
- load config
- resolve source with resolve_dataset_source
- prepare paths with prepare_pploner_paths
- read preprocessing config
- call preprocess_vectorized_dataset on pploner_paths.processed_data_dir
- print concise summary:
  backend
  processed_data_dir
  input X path
  output X_preprocessed path
  y path or none
  n_rows
  n_features
  method

Fail clearly if vectorization has not been run:
"No vectorized X.npy found. Run scripts/vectorize_collide.py first."

3. Update configs.

Add to configs/paths/local.yaml and configs/paths/hf_collide1m.yaml:

preprocessing:
  method: standardize
  eps: 1.0e-6
  input_subdir: vectorized
  output_subdir: preprocessed
  copy_labels: true

Optionally add same block to configs/paths/eos.yaml, but keep EOS safe.

4. Add tests.

Create:

tests/test_preprocessing.py

Tests:
- create temporary processed_data_dir/vectorized/X.npy
- run preprocess_vectorized_dataset
- assert X_preprocessed.npy exists
- assert preprocessing_stats.json exists
- assert preprocessing_manifest.json exists
- assert output mean is approximately 0 for non-constant features
- assert constant feature does not produce NaN or inf
- test y.npy is copied when present
- test missing X.npy raises FileNotFoundError
- test unsupported method raises ValueError

5. Update README.md.

Add local pipeline:

python scripts/make_tiny_parquet_sample.py --config configs/paths/local.yaml
python scripts/prepare_collide_source.py --config configs/paths/local.yaml
python scripts/inspect_dataset.py --config configs/paths/local.yaml
python scripts/vectorize_collide.py --config configs/paths/local.yaml
python scripts/preprocess_collide.py --config configs/paths/local.yaml

Add HF pipeline:

python scripts/prepare_collide_source.py --config configs/paths/hf_collide1m.yaml
python scripts/inspect_dataset.py --config configs/paths/hf_collide1m.yaml
python scripts/vectorize_collide.py --config configs/paths/hf_collide1m.yaml
python scripts/preprocess_collide.py --config configs/paths/hf_collide1m.yaml

6. Update docs.

Create or update:

docs/preprocessing.md

Explain:
- B1.8 implements minimal standardization after vectorization.
- Current scaler uses all available vectorized rows.
- Later, scaler should be fitted only on train split.
- Outputs are in processed_data_dir/preprocessed.
- This prepares the dataset for the next step: minimal classification training.

Update docs/vectorization.md with:
"Next step: preprocessing."

7. Git hygiene.

Ensure .gitignore excludes:
data/
outputs/
*.npy
*.pt
*.pth
*.ckpt
*.parquet

Do not commit generated data.

Definition of done:
- pytest tests/ passes
- local pipeline works:
  python scripts/make_tiny_parquet_sample.py --config configs/paths/local.yaml
  python scripts/prepare_collide_source.py --config configs/paths/local.yaml
  python scripts/vectorize_collide.py --config configs/paths/local.yaml
  python scripts/preprocess_collide.py --config configs/paths/local.yaml
- X_preprocessed.npy is produced under data/processed/collide2v_tiny/preprocessed
- preprocessing_stats.json is produced
- preprocessing_manifest.json is produced
- y.npy is available in preprocessed if labels exist
- no generated data or .npy files are tracked by git
