#!/usr/bin/env python
"""Smoke CLI for future quick plots."""

from __future__ import annotations

import argparse

from flowyforge.core.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe quick plots without reading data.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config.get("data", {})
    print(f"Would make quick plots using config: {args.config}")
    print(f"Data backend: {data.get('backend', '<unknown>')}")


if __name__ == "__main__":
    main()

