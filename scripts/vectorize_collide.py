#!/usr/bin/env python
"""Run minimal COLLIDE parquet-to-numpy vectorization."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source
from flowyforge.data_plugins.collide_v2.vectorization import (
    vectorization_config_from_dict,
    vectorize_parquet_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize tiny COLLIDE parquet samples.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        vector_config = vectorization_config_from_dict(config)
        source = resolve_dataset_source(config)
        if not source.parquet_files:
            raise FileNotFoundError(
                "No parquet files found. Run make_tiny_parquet_sample.py or "
                "prepare_collide_source.py first."
            )
        if source.backend == "eos" and (
            vector_config.max_files is None or vector_config.max_files > 10
        ):
            raise RuntimeError("Refusing large EOS vectorization without explicit small max_files.")

        pipeline_paths = prepare_pploner_paths(source)
        output_dir = pipeline_paths.processed_data_dir / vector_config.output_subdir
        result = vectorize_parquet_files(
            parquet_files=source.parquet_files,
            output_dir=output_dir,
            config=vector_config,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail without a traceback.
        print(f"COLLIDE vectorization failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {source.backend}")
    print(f"Dataset dir: {source.dataset_dir}")
    print(f"Parquet files available: {len(source.parquet_files)}")
    print(f"Output dir: {result.output_dir}")
    print(f"X.npy: {result.x_path}")
    print(f"y.npy: {result.y_path if result.y_path is not None else 'none'}")
    print(f"Rows: {result.n_rows}")
    print(f"Features: {result.n_features}")
    print(f"Feature columns: {', '.join(result.feature_columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
