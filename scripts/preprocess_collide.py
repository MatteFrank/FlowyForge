#!/usr/bin/env python
"""Run minimal COLLIDE vectorized-data preprocessing."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_config
from flowyforge.data_plugins.collide_v2.pploner_adapter import prepare_pploner_paths
from flowyforge.data_plugins.collide_v2.preprocessing import (
    preprocess_vectorized_dataset,
    preprocessing_config_from_dict,
)
from flowyforge.data_plugins.collide_v2.source_resolver import resolve_dataset_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess vectorized COLLIDE arrays.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        source = resolve_dataset_source(config)
        pipeline_paths = prepare_pploner_paths(source)
        preprocessing_config = preprocessing_config_from_dict(config)
        result = preprocess_vectorized_dataset(
            processed_data_dir=pipeline_paths.processed_data_dir,
            config=preprocessing_config,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should fail without a traceback.
        print(f"COLLIDE preprocessing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Backend: {source.backend}")
    print(f"Processed data dir: {pipeline_paths.processed_data_dir}")
    print(f"Input X.npy: {result.x_input_path}")
    print(f"Output X_preprocessed.npy: {result.x_output_path}")
    print(f"y.npy: {result.y_output_path if result.y_output_path is not None else 'none'}")
    print(f"Rows: {result.n_rows}")
    print(f"Features: {result.n_features}")
    print(f"Method: {result.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

