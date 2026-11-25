#!/usr/bin/env python
"""Prepare raw COLLIDE samples: legacy HF smoke or explicit local/HF file pilot."""

from __future__ import annotations

import argparse
import json
import sys

from flowyforge.core.config import load_yaml
from flowyforge.data_plugins.collide_v2.hf_collide1m import prepare_hf_source
from flowyforge.data_plugins.collide_v2.sample_preparation import (
    inspect_prepared_sample, load_sample_config, plan_sample, prepare_sample, prepare_partitions,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paths/hf_collide1m_raw.yaml")
    parser.add_argument("--stage", choices=("plan", "prepare", "partitions", "inspect"), default="prepare",
                        help="Selected files: plan; prepare + partitions; partitions on a completed sample (no export); inspect both views.")
    args = parser.parse_args()
    try:
        config = load_yaml(args.config)
        if "source" in config:
            config = load_sample_config(args.config)
            if args.stage == "plan":
                print(json.dumps(plan_sample(config), indent=2))
            elif args.stage == "partitions":
                manifest = prepare_partitions(config)
                print(f"Partitions: {config['output_dir']}/partitions.json")
                print(f"Integrity: {config['partitions']['check_integrity_SHA256']}; splits: {manifest['split_counts']}")
            elif args.stage == "inspect":
                reports = inspect_prepared_sample(config["output_dir"])
                print(json.dumps(reports, indent=2))
            else:
                manifest = prepare_sample(config)
                print(f"Validated two-view raw sample: {config['output_dir']}")
                print(f"Processes: {manifest['processes']}")
                print("Next: run_classification.py --config configs/runs/collide_pilot.yaml --stage inspect")
            return 0
        if args.stage != "prepare":
            raise ValueError("plan/partitions/inspect require a selected-file pilot config; the legacy HF smoke supports prepare only.")
        manifest = prepare_hf_source(config)
    except Exception as exc:  # Include remote HTTP errors; preserve diagnostics in the manifest.
        print(f"Raw COLLIDE preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Raw nested Parquet: {config['paths']['dataset_dir']}")
    print(f"Rows per process: {manifest['rows_written_per_process']}")
    print("Next: run_classification.py --config configs/runs/hf.yaml --stage inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
