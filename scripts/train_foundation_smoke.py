#!/usr/bin/env python
"""Train the structured COLLIDE object-token/Perceiver feasibility slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from flowyforge.core.config import load_config
from flowyforge.training.foundation_smoke import train_foundation_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--process", action="append", required=True, metavar="NAME=PATH_OR_GLOB",
                        help="Repeat for each class; quote globs. Original nested Parquet only.")
    parser.add_argument("--view", choices=("FullReco", "L1T"), default="FullReco")
    parser.add_argument("--model-config", default="configs/models/foundation_tiny.yaml")
    parser.add_argument("--max-events-per-class", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        processes: dict[str, str] = {}
        for entry in args.process:
            name, separator, path = entry.partition("=")
            name, path = name.strip(), path.strip()
            if not separator or not name or not path:
                raise ValueError(f"Expected --process NAME=PATH_OR_GLOB, got {entry!r}.")
            if name in processes:
                raise ValueError(f"Duplicate process name: {name}")
            processes[name] = path
        # Small attention kernels are faster and more reproducible on one CPU thread.
        torch.set_num_threads(1)
        metrics = train_foundation_smoke(
            processes, view=args.view, model_config=load_config(args.model_config),
            max_events_per_class=args.max_events_per_class, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=args.seed, output_dir=args.output_dir,
        )
    except (ValueError, TypeError, OSError, RuntimeError, FloatingPointError) as exc:
        print(f"Foundation smoke failed: {exc}", file=sys.stderr)
        return 1
    print(f"Files read: {json.dumps(metrics['files_read'])}")
    print(f"Splits: {metrics['split_sizes']}")
    print(f"Diagnostics: {json.dumps(metrics['diagnostics'])}")
    print(f"Best epoch: {metrics['best_epoch']}; validation accuracy: {metrics['validation']['accuracy']:.4f}; "
          f"test accuracy: {metrics['test']['accuracy']:.4f}")
    for warning in metrics["warnings"]:
        print(f"Warning: {warning}")
    print(f"Outputs: {Path(args.output_dir).expanduser() / 'metrics.json'}, "
          f"{Path(args.output_dir).expanduser() / 'checkpoint.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
