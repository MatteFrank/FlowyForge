#!/usr/bin/env python
"""Inspect COLLIDE-2V parquet dataset schemas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.schema_inspector import (
    inspect_dataset_schema,
    write_schema_report,
)
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect parquet schemas without reading row data.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--output-json",
        default="outputs/schema/schema_report.json",
        help="Path for the JSON schema report.",
    )
    parser.add_argument(
        "--output-txt",
        default="outputs/schema/schema_report.txt",
        help="Path for the plain-text schema report.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional override for data.max_files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        data = config.get("data", {})
        source = resolve_dataset_source(config)
        backend = source.backend
        base_path = source.dataset_dir
        max_files = args.max_files if args.max_files is not None else data.get("max_files")
        if max_files is not None and max_files < 0:
            raise ValueError("max_files must be non-negative or None.")

        parquet_files = source.parquet_files
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under base path: {base_path}")

        files_to_inspect = parquet_files[:max_files] if max_files is not None else parquet_files
        report = inspect_dataset_schema(files_to_inspect)
        write_schema_report(
            report,
            output_json=Path(args.output_json),
            output_txt=Path(args.output_txt) if args.output_txt else None,
        )

    except Exception as exc:  # noqa: BLE001 - CLI should fail without a traceback.
        print(f"Dataset inspection failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {backend}")
    print(f"Base path: {base_path}")
    print(f"Parquet files found: {len(parquet_files)}")
    print(f"Files inspected: {report['num_files_inspected']}")
    print(f"Union columns: {len(report['union_columns'])}")
    print(f"JSON report: {args.output_json}")
    if args.output_txt:
        print(f"Text report: {args.output_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
