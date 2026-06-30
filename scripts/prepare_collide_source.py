#!/usr/bin/env python
"""Prepare COLLIDE-2V data-source manifests for downstream pipeline steps."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.hf_collide1m import (
    materialize_hf_collide1m_multi_process_subset,
    materialize_hf_collide1m_subset,
)
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
            if not source.hf_data_dirs and source.hf_data_files is None:
                raise ValueError("HF materialization requires paths.hf_data_dirs or paths.hf_data_files.")
            max_rows_per_process = _max_rows_per_process(data)
            max_files_per_process = _int_config(data, "max_files", 1)
            if len(source.hf_data_dirs) > 1:
                hf_manifest = materialize_hf_collide1m_multi_process_subset(
                    dataset_name=source.hf_dataset_name,
                    split=source.hf_split,
                    output_dir=source.dataset_dir,
                    data_dirs=source.hf_data_dirs,
                    max_rows_per_process=max_rows_per_process,
                    max_files_per_process=max_files_per_process,
                    seed=args.seed,
                    columns=data.get("hf_summary_columns"),
                    materialization_mode=str(data.get("hf_materialization_mode", "summary")),
                )
            else:
                hf_manifest = materialize_hf_collide1m_subset(
                    dataset_name=source.hf_dataset_name,
                    split=source.hf_split,
                    output_dir=source.dataset_dir,
                    max_rows=max_rows_per_process,
                    max_files=max_files_per_process,
                    seed=args.seed,
                    data_dir=source.hf_data_dir,
                    data_files=source.hf_data_files,
                    columns=data.get("hf_summary_columns"),
                    materialization_mode=str(data.get("hf_materialization_mode", "summary")),
                    process_id=0 if source.hf_data_dir else None,
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
        print(f"HF processes requested: {len(source.hf_data_dirs) or 1}")
        print(f"HF data_dirs: {', '.join(source.hf_data_dirs) if source.hf_data_dirs else '<all>'}")
        print(f"HF materialization mode: {data.get('hf_materialization_mode', 'summary')}")
    print(f"Parquet files: {len(source.parquet_files)}")
    print(f"Rows counted: {counts['total_rows']}")
    print(f"File event counts: {pipeline_paths.file_event_counts_path}")
    print(f"Split manifest: {pipeline_paths.split_manifest_path}")
    if hf_manifest is not None:
        print(f"HF processes materialized: {_hf_processes_materialized(hf_manifest)}")
        print(f"HF rows per process: {_hf_rows_by_process(hf_manifest)}")
        print(f"HF classification possible: {_hf_classification_possible(hf_manifest)}")
    return 0


def _int_config(config: dict, key: str, default: int) -> int:
    value = config.get(key)
    return default if value is None else int(value)


def _max_rows_per_process(config: dict) -> int:
    value = config.get("max_rows_per_process")
    if value is not None:
        return int(value)
    return _int_config(config, "max_rows", 20)


def _hf_processes_materialized(manifest: dict) -> int:
    if "processes_materialized" in manifest:
        return int(manifest["processes_materialized"])
    return 1 if int(manifest.get("rows_written", 0)) > 0 else 0


def _hf_rows_by_process(manifest: dict) -> dict[str, int]:
    if "rows_written_per_process" in manifest:
        return {str(key): int(value) for key, value in manifest["rows_written_per_process"].items()}
    data_dir = manifest.get("data_dir") or "<all>"
    return {str(data_dir): int(manifest.get("rows_written", 0))}


def _hf_classification_possible(manifest: dict) -> bool:
    if "classification_possible" in manifest:
        return bool(manifest["classification_possible"])
    return False


if __name__ == "__main__":
    raise SystemExit(main())
