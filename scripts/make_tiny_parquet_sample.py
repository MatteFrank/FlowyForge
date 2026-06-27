#!/usr/bin/env python
"""Create a tiny local COLLIDE-like parquet sample for smoke testing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from flowyforge.core.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny local parquet sample.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--files", type=int, default=None, help="Number of parquet files to create.")
    parser.add_argument("--rows-per-file", type=int, default=8, help="Rows per parquet file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        paths = config.get("paths", {})
        data = config.get("data", {})
        dataset_dir_value = paths.get("dataset_dir") or data.get("base_path")
        if not dataset_dir_value:
            raise ValueError("Config must define paths.dataset_dir or data.base_path.")
        dataset_dir = Path(os.path.expandvars(str(dataset_dir_value))).expanduser()
        num_files = args.files if args.files is not None else int(data.get("max_files") or 2)
        if num_files <= 0:
            raise ValueError("Number of files must be positive.")
        if args.rows_per_file <= 0:
            raise ValueError("rows-per-file must be positive.")

        processes = ["background", "signal"]
        process_ids = {name: index for index, name in enumerate(processes)}
        written: list[Path] = []
        for index in range(num_files):
            process = processes[index % len(processes)]
            process_dir = dataset_dir / process
            process_dir.mkdir(parents=True, exist_ok=True)
            start = index * args.rows_per_file
            frame = pd.DataFrame(
                {
                    "event_id": list(range(start, start + args.rows_per_file)),
                    "pt": [20.0 + index + row for row in range(args.rows_per_file)],
                    "eta": [0.1 * row for row in range(args.rows_per_file)],
                    "process_id": [process_ids[process]] * args.rows_per_file,
                    "process_name": [process] * args.rows_per_file,
                    "view": ["tiny"] * args.rows_per_file,
                }
            )
            output_path = process_dir / f"sample_{index:03d}.parquet"
            frame.to_parquet(output_path, index=False)
            written.append(output_path)

    except Exception as exc:  # noqa: BLE001 - CLI should fail without a traceback.
        print(f"Tiny parquet sample generation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset dir: {dataset_dir}")
    print(f"Parquet files written: {len(written)}")
    for path in written:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
