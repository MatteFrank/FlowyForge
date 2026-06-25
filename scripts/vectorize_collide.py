#!/usr/bin/env python
"""Prepare pploner-compatible paths for future COLLIDE vectorization."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COLLIDE vectorization paths.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        source = resolve_dataset_source(config)
        if not source.parquet_files:
            raise FileNotFoundError(f"No parquet files found under dataset directory: {source.dataset_dir}")
        pipeline_paths = prepare_pploner_paths(source)
    except Exception as exc:  # noqa: BLE001 - CLI should fail without a traceback.
        print(f"COLLIDE vectorization setup failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {source.backend}")
    print(f"Dataset dir: {pipeline_paths.dataset_dir}")
    print(f"Temporary vector data dir: {pipeline_paths.tmp_data_dir}")
    print(f"Processed/vectorized data dir: {pipeline_paths.processed_data_dir}")
    print(f"File event counts path: {pipeline_paths.file_event_counts_path}")
    print(f"Split manifest path: {pipeline_paths.split_manifest_path}")
    print("Vectorization backend prepared; full vectorization implementation is next step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

