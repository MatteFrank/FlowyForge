#!/usr/bin/env python
"""Train tiny FlowyForge task baselines."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source
from flowyforge.training.classification_trainer import train_tiny_mlp_classifier
from flowyforge.training.config import training_config_from_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny task baselines.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--task", default="classification", help="Task name. Currently: classification.")
    parser.add_argument("--model", default="mlp", help="Model name. Currently: mlp.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        source = resolve_dataset_source(config)
        pipeline_paths = prepare_pploner_paths(source)
        training_config = training_config_from_dict(
            config,
            task_override=args.task,
            model_override=args.model,
        )
        if training_config.task != "classification" or training_config.model != "mlp":
            raise ValueError("Only --task classification --model mlp is supported for now.")
        result = train_tiny_mlp_classifier(
            processed_data_dir=pipeline_paths.processed_data_dir,
            config=training_config,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail without traceback.
        print(f"Task training failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {source.backend}")
    print(f"Processed data dir: {pipeline_paths.processed_data_dir}")
    print(f"Output dir: {result.output_dir}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Final train loss: {result.final_train_loss:.6f}")
    print(f"Validation accuracy: {result.val_accuracy if result.val_accuracy is not None else 'none'}")
    print(f"Test accuracy: {result.test_accuracy if result.test_accuracy is not None else 'none'}")
    print(f"Split sizes: {result.n_train} train / {result.n_val} val / {result.n_test} test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
