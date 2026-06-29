#!/usr/bin/env python
"""Evaluate tiny FlowyForge task baselines."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source
from flowyforge.training.evaluation import (
    evaluate_trained_classifier,
    evaluation_config_from_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tiny task baselines.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--task", default="classification", help="Task name. Currently: classification.")
    parser.add_argument("--model", default="mlp", help="Model name. Currently: mlp.")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default=None,
        help="Evaluation split. Defaults to config evaluation.split.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        source = resolve_dataset_source(config)
        pipeline_paths = prepare_pploner_paths(source)
        evaluation_config = evaluation_config_from_dict(
            config,
            task_override=args.task,
            model_override=args.model,
            split_override=args.split,
        )
        if evaluation_config.task != "classification" or evaluation_config.model != "mlp":
            raise ValueError("Only --task classification --model mlp is supported for now.")
        result = evaluate_trained_classifier(
            processed_data_dir=pipeline_paths.processed_data_dir,
            config=evaluation_config,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail without traceback.
        print(f"Task evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {source.backend}")
    print(f"Processed data dir: {pipeline_paths.processed_data_dir}")
    print(f"Split: {result.split}")
    print(f"Samples: {result.n_samples}")
    print(f"Accuracy: {result.accuracy if result.accuracy is not None else 'none'}")
    print(f"Output dir: {result.output_dir}")
    print(f"Report: {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
