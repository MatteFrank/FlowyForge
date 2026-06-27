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

# Status Phase1 testing 

You are continuing development of FlowyForge / M-Fond fast-track COLLIDE-2V.

Current status:
- B1.7 vectorization works on local/HF tiny samples.
- Vectorization produces:
  X.npy
  y.npy when label exists
  feature_map.json
  label_map.json when label exists
  vectorization_manifest.json
- The preserved strategy is pploner-style:
  Parquet dataset -> vectorized .npy files -> preprocessing -> training.

Goal of this task:
Implement B1.8: minimal pploner-style preprocessing for vectorized data.

Do not implement training yet.
Do not implement PyTorch Dataset yet.
Do not implement full EOS-scale preprocessing yet.
Do not break local/HF/EOS backend abstraction.
Do not hardcode /afs paths.
Do not commit generated .npy files.

Main behavior:
Read vectorized X.npy and optional y.npy.
Compute standardization statistics.
Write preprocessed arrays and metadata.

Required output:
processed_data_dir/vectorized/
  X.npy
  y.npy optional
  feature_map.json
  label_map.json optional
  vectorization_manifest.json

processed_data_dir/preprocessed/
  X_preprocessed.npy
  y.npy optional copied or referenced
  preprocessing_stats.json
  preprocessing_manifest.json
  feature_map.json copied
  label_map.json copied if available

1. Add module:

src/flowyforge/data_plugins/collide_v2/preprocessing.py

Implement dataclasses:

@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    method: str
    eps: float
    input_subdir: str
    output_subdir: str
    copy_labels: bool

Defaults:
    method: "standardize"
    eps: 1e-6
    input_subdir: "vectorized"
    output_subdir: "preprocessed"
    copy_labels: true

@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    input_dir: Path
    output_dir: Path
    x_input_path: Path
    x_output_path: Path
    y_output_path: Path | None
    stats_path: Path
    manifest_path: Path
    n_rows: int
    n_features: int
    method: str

Functions:

- preprocessing_config_from_dict(cfg: dict) -> PreprocessingConfig

Read from cfg["preprocessing"] if present, otherwise defaults.

- compute_standardization_stats(X: np.ndarray, eps: float = 1e-6) -> dict

Return:
{
  "mean": [...],
  "std": [...],
  "safe_std": [...],
  "eps": eps,
  "n_rows": int,
  "n_features": int
}

Rules:
- mean and std computed per feature.
- if std < eps, safe_std = 1.0 for that feature.
- keep float values JSON serializable.

- apply_standardization(X: np.ndarray, stats: dict) -> np.ndarray

Return float32 standardized array.

- preprocess_vectorized_dataset(
    processed_data_dir: str | Path,
    config: PreprocessingConfig,
) -> PreprocessingResult

Behavior:
- input_dir = processed_data_dir / config.input_subdir
- output_dir = processed_data_dir / config.output_subdir
- require input_dir/X.npy
- load X.npy
- method must currently be "standardize"; raise clear ValueError otherwise
- compute stats
- write X_preprocessed.npy as float32
- if y.npy exists and copy_labels is true, copy y.npy to output_dir/y.npy
- copy feature_map.json to output_dir/feature_map.json if present
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