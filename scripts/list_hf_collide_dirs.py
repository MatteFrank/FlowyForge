#!/usr/bin/env python
"""List candidate COLLIDE-1M Hugging Face process folders without downloading data."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from typing import Any


DEFAULT_DATASET = "fastmachinelearning/collide-1m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List HF COLLIDE-1M parquet process folders.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id.")
    parser.add_argument("--max-dirs", type=int, default=30, help="Maximum number of folders to print.")
    return parser.parse_args()


def infer_hf_data_dirs(repo_files: Iterable[str]) -> list[str]:
    """Infer top-level folders that contain parquet files from HF repo file names."""

    data_dirs = set()
    for repo_file in repo_files:
        path = str(repo_file).strip("/")
        if not path.endswith(".parquet") or "/" not in path:
            continue
        data_dirs.add(path.split("/", 1)[0])
    return sorted(data_dirs)


def list_hf_data_dirs(dataset: str, max_dirs: int = 30, api: Any | None = None) -> list[str]:
    if max_dirs < 0:
        raise ValueError("max_dirs must be non-negative.")
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError("Install with: pip install huggingface_hub") from exc
        api = HfApi()

    repo_files = api.list_repo_files(repo_id=dataset, repo_type="dataset")
    return infer_hf_data_dirs(repo_files)[:max_dirs]


def main() -> int:
    args = parse_args()
    try:
        data_dirs = list_hf_data_dirs(args.dataset, max_dirs=args.max_dirs)
    except Exception as exc:  # noqa: BLE001 - CLI should fail without traceback.
        print(f"HF directory listing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset: {args.dataset}")
    print("Candidate data dirs:")
    if not data_dirs:
        print("  <none found>")
        return 0
    for index, data_dir in enumerate(data_dirs, start=1):
        print(f"{index:2d}. {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
