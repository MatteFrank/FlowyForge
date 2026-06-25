"""Safe helpers for materializing small COLLIDE-1M Hugging Face subsets."""

from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd


def materialize_hf_collide1m_subset(
    dataset_name: str,
    split: str,
    output_dir: str | Path,
    max_rows: int = 10000,
    max_files: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """Materialize a small streaming HF subset as local parquet files."""

    if max_rows < 0:
        raise ValueError("max_rows must be non-negative.")
    if max_files <= 0:
        raise ValueError("max_files must be positive.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install with: pip install datasets huggingface_hub") from exc

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    streamed_dataset = load_dataset(dataset_name, split=split, streaming=True)
    row_iter = islice(iter(streamed_dataset), max_rows)
    chunk_size = max(1, math.ceil(max_rows / max_files)) if max_rows else 1

    parquet_files: list[str] = []
    rows_written = 0
    for file_index in range(max_files):
        rows = list(islice(row_iter, chunk_size))
        if not rows:
            break

        frame = pd.DataFrame(rows)
        parquet_path = output_path / f"part-{file_index:05d}.parquet"
        frame.to_parquet(parquet_path, index=False)
        parquet_files.append(parquet_path.name)
        rows_written += len(frame)

        if rows_written >= max_rows:
            break

    manifest = {
        "source_dataset_name": dataset_name,
        "split": split,
        "seed": seed,
        "max_rows": max_rows,
        "max_files": max_files,
        "rows_written": rows_written,
        "parquet_files": parquet_files,
        "warning": "This is a streaming subset, not the full COLLIDE-1M dataset.",
    }
    manifest_path = output_path / "hf_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


__all__ = ["materialize_hf_collide1m_subset"]

