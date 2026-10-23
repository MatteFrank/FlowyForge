# Structured COLLIDE foundation smoke

This is the first supervised feasibility slice:

```text
Original nested COLLIDE-V2 Parquet (one row per event)
  -> EventBatch with heterogeneous ObjectGroups and validity masks
  -> ObjectTokenizer (separate projections + learned collection embeddings)
  -> Perceiver (learned latents cross-attend to objects, then self-attend)
  -> mean latent pooling + output projection
  -> shared event embedding
  -> classification head -> process logits
```

**Use original nested Parquet. The scalar-summary Parquet exported for B1,
and B1's X.npy/y.npy, are not inputs to this pipeline.** B1 commands and code
remain separate. No download, EOS query, or external service is required by the
tests. Real COLLIDE-V2 validation is deferred to the user's LXPLUS run.

## Installation and local checks

From the repository root, use an environment containing the existing package
dependencies (`torch`, `pyarrow`, `numpy`, `pyyaml`, and `pytest`):

```bash
python -m pip install -e .
python -m pytest -q tests/test_foundation_vertical_slice.py tests/test_model_plugins.py
python -m pytest -q tests/
git diff --check
```

Tests create synthetic Parquet in temporary directories and run two short CLI
trainings, covering both views and checkpoint reload. They do not use real
datasets, network, CERNBox, or EOS. The synthetic schema tests the explicit
contract below; it is not evidence that every real production file has that schema.

## Data contract

Each process argument supplies a class name and a local file, directory (searched
recursively), or quoted glob. Mounted EOS paths work as filesystem paths on
LXPLUS; remote `root://`/HTTP access is not implemented. Files are sorted and
deduplicated. Assigning the same resolved file to two classes is an error.

The reader selects exact columns named `{view}_{collection}_{field}`. `view` is
`FullReco` or `L1T`; the API and transformations are otherwise identical.

| Collection | Required fields | Optional fields, zero-filled if absent | Feature width | YAML object cap |
| --- | --- | --- | --- | ---: |
| PUPPIPart | PT, Eta, Phi | PID, Charge | 6 | 128 |
| JetPuppiAK4 | PT, Eta, Phi | Mass | 5 | 16 |
| Electron | PT, Eta, Phi | Charge | 5 | 8 |
| MuonTight | PT, Eta, Phi | Charge | 5 | 8 |
| PhotonTight | PT, Eta, Phi | — | 4 | 8 |
| PUPPIMET | MET, Phi | — | 3 | 1 |

For example, required particle columns include `FullReco_PUPPIPart_PT`,
`FullReco_PUPPIPart_Eta`, and `FullReco_PUPPIPart_Phi`. No `JetAK4` ->
`JetPuppiAK4`, `MET` -> `PUPPIMET`, charge spelling, or other aliases are inferred.
These defaults define an explicit expected contract, validated against each file
actually read. Original production schemas were not available in the local
workspace; confirm the exact fields and structure on LXPLUS before interpreting
results. A missing required column is an error that includes the file and column.

Each field is a numeric object list. A singleton outer list around that vector
is accepted, such as `[[10, 20]]`; multiple nested vectors are rejected rather
than flattened. Empty lists and null collection lists represent zero objects.
All present fields in a collection must have equal lengths, including optional
fields. A null optional list is treated as unavailable. A null scalar inside a
list is treated as a non-finite numeric value and replaced with zero.

The initial features are, in order:

- `PT`, `MET`, `Mass`: nonnegative clipping, cap at `value_clip`, then `log1p`.
- `Eta`: clip to `[-eta_clip, eta_clip]`, then divide by `eta_clip`.
- `Phi`: sine and cosine (two features).
- `PID`: signed `log1p(abs(PID))`, capped by `value_clip`.
- `Charge`: clip to `[-3, 3]`.

NaN and positive/negative infinity become zero before transformation; counters
record replacements. Real objects remain valid even when a numeric value was
repaired. Objects are sorted by descending nonnegative PT before truncation;
MET preserves source order. Sorting is stable for equal PT. Metadata records
source columns, feature order, original/retained/truncated counts, and missing
optional fields. Original source file and row are retained for each event.

## Configuration and shapes

`configs/models/foundation_tiny.yaml` configures the model and data caps. Feature
widths derive from the collection specs rather than a second dimensions table
in YAML. A collection limit of zero explicitly disables that collection; omitted
limits leave it uncapped. An enabled collection with absent required columns is
an error; an existing empty collection is valid. Do not disable missing
collections without first checking whether the correct original files were supplied.

Defaults: embedding width 64, 16 learned latents, 4 attention heads, 2 latent
self-attention blocks, feed-forward multiplier 4, dropout 0.1, output width 64.
There is one initial cross-attention block plus its feed-forward block. Latent
blocks also have residual connections, layer normalization and feed-forward layers.

| Stage | Shape with defaults |
| --- | --- |
| Group features | `[B, N_g, F_g]`, float32 |
| Group validity mask | `[B, N_g]`, bool; True means a real object |
| Concatenated tokens | `[B, sum_g(N_g), 64]` |
| Token mask / group IDs | `[B, sum_g(N_g)]`, bool / int64 |
| Latents | `[B, 16, 64]` |
| Shared embedding | `[B, 64]` |
| Classification logits | `[B, C]` |
| Targets | `[B]`, int64 |

Each row initially has `B=1` and only retained objects. Collation pads each
collection to the longest sequence in that batch, so default allocated token
count is at most 169 per event. Actual valid token counts come from masks.
Group IDs remain fixed in configured collection order, including for padding;
the mask determines validity. The tokenizer accepts subsets of configured
collections and rejects unknown groups or incorrect feature widths.

Masked values are removed before projection and cannot affect attention. An
empty event can attend to a zero sentinel inside the Perceiver; this prevents
all-masked attention NaNs. The sentinel is hidden from nonempty events and is
not part of tokenizer counts. Empty-event predictions use learned latent priors.
An entirely empty `EventBatch` without groups needs `metadata['batch_size']`.

`FoundationEncoder` implements `ModelPlugin.encode`, `forward`, and
`get_embeddings`. `EventClassificationModel` holds the supplied encoder instance
and adds LayerNorm -> Linear -> GELU -> Linear. Its `encode` method returns the
same shared representation; its `forward` returns logits.

## Manual FullReco run on LXPLUS

From the repository root with the package installed, set the two process paths
to your **original nested** files. The root below comes from the existing EOS
config; the process subdirectory layout must be confirmed on your installation.
This command is provided for manual execution and has not been run against EOS.

```bash
export COLLIDE_ROOT=/eos/project/f/foundational-model-dataset/samples/production_final
python scripts/train_foundation_smoke.py \
  --process "WJets=${COLLIDE_ROOT}/WJetsToLNu_13TeV-madgraphMLM-pythia8" \
  --process "DYJets=${COLLIDE_ROOT}/DYJetsToLL_13TeV-madgraphMLM-pythia8" \
  --view FullReco \
  --model-config configs/models/foundation_tiny.yaml \
  --max-events-per-class 128 \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --seed 42 \
  --output-dir outputs/foundation_smoke/fullreco
```

## Manual L1T run on LXPLUS

The same original files may hold both views. Otherwise change the two input
paths to the L1T-containing files. Required `L1T_...` columns are checked with
the same contract, without translating FullReco objects into L1T objects.

```bash
export COLLIDE_ROOT=/eos/project/f/foundational-model-dataset/samples/production_final
python scripts/train_foundation_smoke.py \
  --process "WJets=${COLLIDE_ROOT}/WJetsToLNu_13TeV-madgraphMLM-pythia8" \
  --process "DYJets=${COLLIDE_ROOT}/DYJetsToLL_13TeV-madgraphMLM-pythia8" \
  --view L1T \
  --model-config configs/models/foundation_tiny.yaml \
  --max-events-per-class 128 \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --seed 42 \
  --output-dir outputs/foundation_smoke/l1t
```

## Training and outputs

At least two processes and three events per process are required. Class IDs use
sorted process names. Loading takes the first capped rows from sorted files.
A seeded, disjoint split allocates approximately 60/20/20 per class, preserving
at least one event per class in every split. Very small classes use 1/1/1.

The CPU smoke uses Adam, cross-entropy, gradient clipping at norm 1, and checks
for finite logits, loss, gradients and parameters. It trains the tokenizer,
Perceiver and head together. The CLI uses one CPU thread. It selects the
checkpoint with minimum validation loss; the test set is evaluated only after
selection. There is no fitted dataset-wide normalization or test-based selection.

The output directory contains:

- `metrics.json`: input file matches, files actually contributing rows, source
  row provenance, label mapping, effective config, split indices/class counts,
  per-epoch train loss and validation metrics, best epoch, selected-model
  train/validation/test loss, accuracy and confusion matrices, shape/mask/token
  diagnostics, replacement/truncation counts and warnings. Confusion matrices
  use true labels as rows and predicted labels as columns.
- `checkpoint.pt`: selected model state, effective config, ordered collection
  dimensions/specs, labels, run arguments, best epoch and split indices.
  It is an inference checkpoint, without optimizer state or resume support.

Use a different output directory for each run; rerunning in the same directory
overwrites those two artifacts. Existing `.gitignore` rules exclude outputs and
checkpoint/data formats.

To reconstruct the model (with the same feature transformations):

```python
import torch
from flowyforge.model_plugins import (
    EventClassificationModel, FoundationEncoder, PerceiverConfig,
)

checkpoint = torch.load("outputs/foundation_smoke/fullreco/checkpoint.pt",
                        map_location="cpu", weights_only=True)
config = dict(checkpoint["config"]["model"])
config.pop("name")
model = EventClassificationModel(
    FoundationEncoder(checkpoint["collection_feature_dims"], PerceiverConfig(**config)),
    num_classes=len(checkpoint["label_map"]),
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## What to check and current limits

Check `files_read` and `event_sources` first, then split counts for every class.
In `diagnostics`, inspect collection feature widths, actual retained counts,
padding masks, truncated totals, optional missing fields and non-finite input
replacements. Sample shapes refer to the first loaded batch; collection totals
cover the entire loaded sample. Check token counts against the sum of valid
objects, embedding `[B,64]`, logits `[B,C]`, both finite flags and finite gradients.
Review epoch train loss, validation loss/accuracy and the selected test metrics.
Loss need not decrease monotonically, and tiny-sample accuracy is not a benchmark.

This is bounded, in-memory training on CPU; it is not streaming training or
full-scale vectorization. Sampling the first rows can bias the result. Splits
are by row, not by file or generator identity; duplicated physics events across
distinct files are not detected. No luminosity weights, cross sections, class
balancing or scientific validation are included. Optional features are zero
imputed without a separate feature-presence channel. PID is a numeric smoke
feature, not a learned categorical particle vocabulary. There are no position
embeddings: the encoder treats each collection as a set.

Production schema compatibility, rare object formats and physics performance
remain to be validated on LXPLUS. Graphs/GNNs, self-supervision, masked modeling,
joint Offline-L1T losses, JEPA, contrastive learning, MoE, distributed training,
scaling and full physics benchmarks are outside this slice.
