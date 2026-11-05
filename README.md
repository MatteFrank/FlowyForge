# FlowyForge / M-Fond

FlowyForge is a modular foundation platform for complex scientific data. The first fast-track path targets COLLIDE-2V / HEP workflows while keeping the package extensible to future datasets, task plugins, model plugins, and analysis tools.

Current status: COLLIDE source inspection, the B1 scalar-summary/MLP baseline, and structured object classification (ObjectTokenizer -> latent Perceiver -> event embedding -> classification). A shared streaming runner supports small original Parquet samples and full mounted EOS datasets, with CPU/GPU execution and reports. Real COLLIDE-V2 schema/unit validation and full-scale runs remain to be performed on an authorized CERN node.

## Install

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

The requirements file installs portable Python dependencies, including PyTorch,
pytest and the Hugging Face backend. The final command installs FlowyForge in
editable mode so scripts can import the package under `src/`.
For only the core dependencies, use `python -m pip install -e .` instead.

### Linux, LXPLUS and HPC environments

Create the virtual environment on the target system; do not copy a macOS venv
to Linux or reuse one across incompatible cluster environments. If your site
uses environment modules, load its Python environment before creating the venv,
and use the same environment in batch jobs. Module names and available Python
versions depend on the site.

`requirements.txt` contains no local paths, macOS-only packages, CUDA version or
custom package index. It can also be installed before the repository is available;
run the editable project installation once you are in the repository root.
This is a portable dependency list, not a lock file validated on every cluster.

For GPU use, install the PyTorch build prescribed by your cluster for its driver
and accelerator environment **before** installing the requirements. The plain
`torch` requirement accepts an already installed build; it does not select or
verify a CUDA version. The foundation smoke trainer currently runs on CPU even
when CUDA is available. Installing a GPU build does not change that behavior.

Check the environment on the node where you will run the workload:

```bash
python -m pip check
python -c "import flowyforge, torch, pyarrow; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('PyArrow:', pyarrow.__version__)"
python -m pytest -q tests/
```

A CPU-only node may correctly report `CUDA available: False`. On clusters
without outbound network access, install through the site's package mirror or
an approved wheel directory; dependency installation needs to be completed
before running offline jobs. EOS access and filesystem mounts are supplied by
the cluster, not by this requirements file.

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
- mounted EOS paths on LXPLUS (streaming structured classification; B1 vectorization remains separate)

Docs:

- [Pipeline explained](docs/pipeline_explained.md)
- [Vectorization](docs/vectorization.md)
- [Preprocessing](docs/preprocessing.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [HF smoke pipeline](docs/hf_smoke_pipeline.md)
- [HF multi-process classification](docs/hf_multi_process_classification.md)
- [Structured foundation smoke: data contract, FullReco/L1T commands, outputs](docs/foundation_smoke.md)
- [Small/EOS classification: preflight, pilot, full GPU run and reports](docs/classification_run.md)

## Streaming Classification

Use `scripts/run_classification.py` with `configs/runs/classification.yaml` for
both `--mode small` and `--mode eos`. The stages `inspect`, `train`, `evaluate`
and `report` cover schema/unit auditing, uniform sampling across files, bounded
batch construction, CPU/GPU execution and reproducible evaluation. Outputs
include CSV tables, PNG/PDF figures and a self-contained HTML report.

The EOS root defaults to `/eos/project/f/foundational-model-dataset/samples/`.
Start with `--stage inspect`; see the [detailed run guide](docs/classification_run.md)
for exact commands, input contracts, memory controls and the limitations of
preliminary unweighted results. The existing smoke and B1 paths below remain available.

## Structured Foundation Smoke

This path reads **original nested COLLIDE-V2 Parquet**, not the scalar-summary
Parquet or vectorized arrays produced by B1. The existing B1 commands below
remain unchanged. Both `--view FullReco` and `--view L1T` use the same object API;
required column names and shapes are checked explicitly, without inferred aliases.

```bash
python scripts/train_foundation_smoke.py \
  --process 'signal=/path/to/original/signal/*.parquet' \
  --process 'background=/path/to/original/background/*.parquet' \
  --view FullReco \
  --model-config configs/models/foundation_tiny.yaml \
  --max-events-per-class 128 --epochs 3 --batch-size 16 \
  --learning-rate 0.001 --seed 42 \
  --output-dir outputs/foundation_smoke/fullreco
```

Replace the process paths with original nested files. Use `--view L1T` and a
separate output directory for the other view. The CPU smoke saves `metrics.json`
and `checkpoint.pt`; it needs at least two classes and three events per class.
See the [foundation smoke guide](docs/foundation_smoke.md) for the exact schema,
collection limits, diagnostics and manual LXPLUS commands.

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

## B1 EOS Inspection / Vectorization

```bash
python scripts/inspect_dataset.py --config configs/paths/eos.yaml
python scripts/vectorize_collide.py --config configs/paths/eos.yaml
```

The B1 pploner-style flow is preserved as `Parquet -> vectorized .npy -> preprocessing -> training`.
For structured streaming classification on the full EOS dataset, use the new
[classification runner](docs/classification_run.md).
