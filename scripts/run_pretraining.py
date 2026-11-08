#!/usr/bin/env python
"""Inspect or train masked COLLIDE reconstruction using existing shared partitions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runs/collide_pretraining.yaml")
    parser.add_argument("--stage", choices=("inspect", "train"), default="train")
    parser.add_argument("--resume", help="Last checkpoint.pt in the same output directory; epochs is the total horizon.")
    parser.add_argument("--view", choices=("FullReco", "L1T"))
    parser.add_argument("--output-dir")
    parser.add_argument("--partition-manifest")
    parser.add_argument("--model-config")
    parser.add_argument("--mode", choices=("small", "eos"))
    parser.add_argument("--data-root")
    parser.add_argument("--process", action="append", metavar="NAME=PATH_OR_GLOB")
    parser.add_argument("--process-name", action="append")
    parser.add_argument("--max-events-per-class", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda[:index], or mps")
    for name in ("epochs", "batch-size", "num-workers", "seed"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args()
    try:
        from flowyforge.training.pretraining_run import load_pretraining_config, run_pretraining

        overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k not in {"config", "stage", "resume", "process", "process_name"}}
        if args.process_name:
            overrides["process_names"] = args.process_name
        if args.process:
            processes = {}
            for entry in args.process:
                name, separator, path = entry.partition("=")
                name, path = name.strip(), path.strip()
                if not separator or not name or not path or name in processes:
                    raise ValueError(f"Expected unique NAME=PATH, got {entry!r}.")
                processes[name] = path
            overrides["processes"] = processes
        config = load_pretraining_config(args.config, overrides)
        result = run_pretraining(config, inspect_only=args.stage == "inspect", resume=args.resume)
        output = Path(config["output_dir"])
        if args.stage == "inspect":
            print(f"Preflight: {output / 'preflight.json'}; test excluded.")
        else:
            print(f"Best epoch: {result['best_epoch']}; transferable encoder: {output / 'encoder.pt'}")
            print(f"Resume checkpoint: {output / 'checkpoint.pt'}; metrics: {output / 'metrics.json'}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, FloatingPointError) as exc:
        print(f"Pretraining {args.stage} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
