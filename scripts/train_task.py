#!/usr/bin/env python
"""Smoke CLI for future task training."""

from __future__ import annotations

import argparse

from flowyforge.core.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe training without running ML.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config.get("data", {})
    print(f"Would train task using config: {args.config}")
    print(f"Data backend: {data.get('backend', '<unknown>')}")


if __name__ == "__main__":
    main()

