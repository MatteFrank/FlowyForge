# COLLIDE Training

B1.9 adds the first minimal supervised classification baseline.

This is a pipeline sanity check, not a physics-validated model. It proves that the current local/HF tiny-sample flow can reach a trainable supervised array format:

```text
Parquet -> inspect/manifest -> vectorize -> preprocess -> train tiny classifier
```

## Inputs

Training consumes preprocessed arrays:

```text
processed_data_dir/preprocessed/X_preprocessed.npy
processed_data_dir/preprocessed/y.npy
```

`X_preprocessed.npy` must be a 2D float feature matrix. `y.npy` must be a 1D integer label array with at least two classes.

## Command

```bash
python scripts/train_task.py --config configs/paths/local.yaml --task classification --model mlp
```

Only `classification` with `mlp` is currently supported.

## Model

The baseline model is a tiny PyTorch MLP:

```text
Linear(input_dim, hidden_dim)
ReLU
optional Dropout
Linear(hidden_dim, n_classes)
```

It returns logits and is trained with `torch.nn.CrossEntropyLoss`.

## Outputs

Training writes under:

```text
processed_data_dir/training/classification_mlp/
```

Files:

- `checkpoint.pt`
- `metrics.json`
- `training_manifest.json`

`checkpoint.pt` contains the model state dict, model dimensions, feature map, label map, and training config.

`metrics.json` contains final train loss, optional validation/test accuracy, split sizes, number of features, number of classes, epochs, and learning rate.

`training_manifest.json` records the processed data directory, input subdirectory, output directory, checkpoint path, metrics path, task, model, and a warning that this is a minimal tiny-sample classifier.

## Current Limitations

- Not EOS-scale.
- Not physics-validated.
- Uses a simple random train/val/test split.
- Preprocessing statistics are not yet fit on train only.
- No class balancing yet.
- No PyTorch Lightning, Hydra, MLflow, or Optuna.
- No trigger task, transformer baseline, graph model, or foundation-ready model training yet.

Future training work may add train-only preprocessing statistics, better splits, class balancing, richer task/model plugins, trigger training, transformer baselines, and graph/foundation-ready models.

