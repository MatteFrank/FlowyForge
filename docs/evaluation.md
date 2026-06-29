# COLLIDE Evaluation

B1.10 adds minimal evaluation and reporting for the tiny MLP classification sanity check.

This is not physics validation. It only checks that the current end-to-end path can train a small model and reload it for deterministic split-based evaluation.

## Inputs

Evaluation consumes:

```text
processed_data_dir/preprocessed/X_preprocessed.npy
processed_data_dir/preprocessed/y.npy
processed_data_dir/training/classification_mlp/checkpoint.pt
```

The checkpoint stores the training config. Evaluation uses that config to reproduce the same train/val/test split through `val_fraction`, `test_fraction`, and `seed`.

## Command

```bash
python scripts/evaluate_task.py --config configs/paths/local.yaml --task classification --model mlp
```

Optional split selection:

```bash
python scripts/evaluate_task.py --config configs/paths/local.yaml --task classification --model mlp --split all
```

Supported splits are `train`, `val`, `test`, and `all`.

## Outputs

Evaluation writes under:

```text
processed_data_dir/evaluation/classification_mlp/
```

Files:

- `evaluation_metrics.json`
- `confusion_matrix.json`
- `predictions.csv`
- `evaluation_report.md`

`evaluation_metrics.json` contains accuracy, macro accuracy, sample count, class count, per-class support/correct/accuracy, checkpoint path, and the tiny-sample warning.

`confusion_matrix.json` stores a confusion matrix with rows as true classes and columns as predicted classes.

`predictions.csv` contains:

```text
index,y_true,y_pred,correct
```

`evaluation_report.md` is a short human-readable report with split, sample count, metrics, output paths, and the tiny sanity-check warning.

## Current Limitations

- Not EOS-scale.
- Not physics-validated.
- No advanced physics metrics.
- No plotting yet.
- No class-balancing analysis yet.
- No uncertainty or calibration metrics.

Future evaluation can add richer classification reports, plots, physics-derived metrics, split-aware comparisons, and integration with more mature experiment tracking if useful.

