#!/usr/bin/env python
"""Prepare COLLIDE-2V data-source manifests for downstream pipeline steps."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.hf_collide1m import materialize_hf_collide1m_subset
from flowyforge.data_plugins.collide_v2.manifest import (
    create_split_manifest,
    scan_parquet_event_counts,
    write_file_event_counts,
    write_split_manifest,
)
from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COLLIDE source manifests.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic split manifests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        data = config.get("data", {})
        source = resolve_dataset_source(config)

        hf_manifest = None
        if source.backend == "hf" and data.get("materialize_local_parquet", False):
            if source.hf_dataset_name is None or source.hf_split is None:
                raise ValueError("HF backend requires paths.hf_dataset_name and paths.hf_split.")
            hf_manifest = materialize_hf_collide1m_subset(
                dataset_name=source.hf_dataset_name,
                split=source.hf_split,
                output_dir=source.dataset_dir,
                max_rows=int(data.get("max_rows") or 20),
                max_files=int(data.get("max_files") or 1),
                seed=args.seed,
                data_dir=source.hf_data_dir,
                data_files=source.hf_data_files,
                columns=data.get("hf_summary_columns"),
                materialization_mode=str(data.get("hf_materialization_mode", "summary")),
            )
            source = resolve_dataset_source(config)

        if not source.parquet_files:
            raise FileNotFoundError(f"No parquet files found under dataset directory: {source.dataset_dir}")

        pipeline_paths = prepare_pploner_paths(source)
        counts = scan_parquet_event_counts(source.dataset_dir)
        write_file_event_counts(counts, pipeline_paths.file_event_counts_path)

        split_manifest = create_split_manifest(
            counts=counts,
            split_counts=data.get("train_val_test_split_per_class"),
            classnames=data.get("to_classify") or data.get("classnames"),
            folder_map=data.get("process_to_folder"),
            seed=args.seed,
        )
        if pipeline_paths.split_manifest_path is not None:
            write_split_manifest(split_manifest, pipeline_paths.split_manifest_path)

    except Exception as exc:  # noqa: BLE001 - CLI should fail without a traceback.
        print(f"COLLIDE source preparation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {source.backend}")
    print(f"Dataset dir: {source.dataset_dir}")
    if source.backend == "hf":
        print(f"HF data_dir: {source.hf_data_dir or '<all>'}")
        print(f"HF materialization mode: {data.get('hf_materialization_mode', 'summary')}")
    print(f"Parquet files: {len(source.parquet_files)}")
    print(f"Rows counted: {counts['total_rows']}")
    print(f"File event counts: {pipeline_paths.file_event_counts_path}")
    print(f"Split manifest: {pipeline_paths.split_manifest_path}")
    if hf_manifest is not None:
        print(f"HF rows materialized: {hf_manifest['rows_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
