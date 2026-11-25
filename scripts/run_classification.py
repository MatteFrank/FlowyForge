#!/usr/bin/env python
"""Create shared partitions, inspect, train, evaluate, report or compare COLLIDE runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/runs/classification.yaml")
    parser.add_argument("--stage", choices=("partitions", "inspect", "train", "evaluate", "report", "compare", "compare-transfer"), default="train")
    parser.add_argument("--run-dir", action="append", help="Repeat for two views (compare) or three downstream modes (compare-transfer).")
    parser.add_argument("--downstream-mode", choices=("scratch", "linear_probe", "finetune"),
                        help="Explicit benchmark with a common linear head; omitted preserves the legacy classifier.")
    parser.add_argument("--encoder-checkpoint", help="Exported encoder.pt, required by linear_probe/finetune.")
    parser.add_argument("--partition-manifest", help="Create strict v1 at --stage partitions; reuse strict v1 / fast v2 for inspect/train. Path relative to cwd.")
    parser.add_argument("--mode", choices=("small", "eos"))
    parser.add_argument("--process", action="append", metavar="NAME=PATH_OR_GLOB",
                        help="Repeat for each class. Overrides the selected YAML process map; quote globs.")
    parser.add_argument("--data-root", help="EOS/local root for process directory discovery.")
    parser.add_argument("--process-name", action="append", help="Select a discovered process; repeat as needed.")
    parser.add_argument("--max-events-per-class", type=int, help="Uniform per-class sample cap; 0 means all events.")
    parser.add_argument("--view", choices=("FullReco", "L1T"))
    parser.add_argument("--model-config", help="Model YAML; CLI paths are relative to the working directory.")
    parser.add_argument("--epochs", type=int, help="Total training horizon, including epochs already completed before --resume.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint", help="For evaluate; default checkpoint.pt in the training output configured in YAML.")
    parser.add_argument("--resume", help="For train: last_checkpoint.pt in the same output directory. --epochs is the TOTAL horizon.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from flowyforge.core.config import load_yaml

    try:
        if args.resume and args.stage != "train":
            raise ValueError("--resume is only supported with --stage train.")
        if args.stage in {"compare", "compare-transfer"}:
            from flowyforge.analysis.classification_report import compare_classification_runs, compare_downstream_runs

            invalid = [key for key, value in vars(args).items()
                       if value is not None and key not in {"stage", "config", "run_dir", "output_dir"}]
            if invalid or not args.run_dir or not args.output_dir:
                raise ValueError(f"{args.stage} requires --run-dir paths and --output-dir; no training/evaluation overrides.")
            compare = compare_classification_runs if args.stage == "compare" else compare_downstream_runs
            compare(args.run_dir, args.output_dir)
            print(f"Verified comparison: {Path(args.output_dir) / 'comparison.html'}")
            return 0
        if args.run_dir:
            raise ValueError("--run-dir is only used by --stage compare or compare-transfer.")
        if args.stage in {"evaluate", "report"}:
            invalid = [name for name in ("mode", "process", "process_name", "max_events_per_class",
                                         "view", "model_config", "epochs", "seed", "partition_manifest", "downstream_mode", "encoder_checkpoint") if getattr(args, name) is not None]
            if args.stage == "report" and args.data_root is not None:
                invalid.append("data_root")
            if invalid:
                raise ValueError(f"{args.stage} uses the saved run; remove input/training overrides: {invalid}")
            if args.stage == "report":
                from flowyforge.analysis.classification_report import write_classification_report

                if args.checkpoint or any(getattr(args, key) is not None for key in ("device", "batch_size", "num_workers")):
                    raise ValueError("report reads metrics.json only; use --output-dir to select the saved run.")
                default_output = args.output_dir or load_yaml(args.config).get("output_dir", "outputs/classification")
                output = Path(default_output).expanduser()
                metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                print(f"Report: {write_classification_report(metrics, output)}")
                return 0
            default_output = None if args.checkpoint else load_yaml(args.config).get("output_dir", "outputs/classification")
            checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else Path(default_output).expanduser() / "checkpoint.pt"
            output = Path(args.output_dir).expanduser() if args.output_dir else checkpoint.parent / "evaluation"
            from flowyforge.training.classification_run import evaluate_run

            metrics = evaluate_run(checkpoint, output_dir=output, device=args.device,
                                   batch_size=args.batch_size, num_workers=args.num_workers, data_root=args.data_root)
        else:
            from flowyforge.training.classification_run import load_run_config, run_classification

            if args.checkpoint:
                raise ValueError("--checkpoint is only used by --stage evaluate.")
            overrides = {key: value for key, value in vars(args).items()
                         if value is not None and key not in {"config", "stage", "process", "process_name", "checkpoint", "run_dir", "resume"}}
            if args.process_name is not None:
                overrides["process_names"] = args.process_name
            if args.process:
                processes = {}
                for entry in args.process:
                    name, separator, value = entry.partition("=")
                    name, value = name.strip(), value.strip()
                    if not separator or not name or not value or name in processes:
                        raise ValueError(f"Expected a unique NAME=PATH process, got {entry!r}.")
                    processes[name] = value
                overrides["processes"] = processes
            config = load_run_config(args.config, overrides)
            if args.stage == "partitions":
                from flowyforge.data_plugins.collide_v2.streaming import create_partition_manifest, save_partition_manifest

                if not config.get("partition_manifest"):
                    raise ValueError("partitions requires --partition-manifest PATH or partition_manifest in YAML.")
                target = Path(config["partition_manifest"]).expanduser()
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite existing partitions: {target}. Reuse them or choose a new path.")
                partitions = create_partition_manifest(config["source"], seed=config["seed"],
                    validation_fraction=config["data"]["validation_fraction"],
                    test_fraction=config["data"]["test_fraction"], read_batch_size=config["data"]["read_batch_size"])
                save_partition_manifest(partitions, target)
                print(f"Partitions: {target}; counts: {partitions['split_counts']}; SHA256: {partitions['sha256']}")
                return 0
            output = Path(config["output_dir"])
            from flowyforge.training.classification_run import resolve_device
            from flowyforge.training.run_statistics import computation_statistics

            with computation_statistics(output, stage=args.stage, resume=args.resume,
                    device=resolve_device(config["training"]["device"] if args.stage == "train" else "cpu"),
                    num_workers=config["training"]["num_workers"]):
                metrics = run_classification(config, inspect_only=args.stage == "inspect", resume=args.resume)
        if args.stage == "inspect":
            print(f"Preflight complete: {output / 'preflight.json'}")
        else:
            print(f"Best epoch: {metrics['best_epoch']}; test accuracy: {metrics['test']['accuracy']:.4f}; "
                  f"macro F1: {metrics['test']['macro_f1']:.4f}")
            print(f"Report: {output / 'report.html'}")
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, FloatingPointError) as exc:
        print(f"Classification {args.stage} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
