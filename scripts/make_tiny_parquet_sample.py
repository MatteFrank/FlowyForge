#!/usr/bin/env python
"""Create nested FullReco/L1T toy data for the same Perceiver runner used on EOS."""

from __future__ import annotations

import argparse
import sys

from flowyforge.data_plugins.collide_v2.synthetic import make_tiny_parquet_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/samples/perceiver_tiny")
    parser.add_argument("--events-per-class", type=int, default=40)
    parser.add_argument("--files-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        paths = make_tiny_parquet_sample(args.output_dir, events_per_class=args.events_per_class,
                                         files_per_class=args.files_per_class, seed=args.seed)
    except (ValueError, OSError) as exc:
        print(f"Nested sample generation failed: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    print("Next: run_classification.py --config configs/runs/local.yaml --stage inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
