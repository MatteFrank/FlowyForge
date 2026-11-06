#!/usr/bin/env python
"""Materialize a bounded raw HF sample; local/EOS Parquet needs no preparation."""

from __future__ import annotations

import argparse
import sys

from flowyforge.core.config import load_yaml
from flowyforge.data_plugins.collide_v2.hf_collide1m import prepare_hf_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paths/hf_collide1m_raw.yaml")
    args = parser.parse_args()
    try:
        config = load_yaml(args.config)
        manifest = prepare_hf_source(config)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError) as exc:
        print(f"Raw HF preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Raw nested Parquet: {config['paths']['dataset_dir']}")
    print(f"Rows per process: {manifest['rows_written_per_process']}")
    print("Next: run_classification.py --config configs/runs/hf.yaml --stage inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
