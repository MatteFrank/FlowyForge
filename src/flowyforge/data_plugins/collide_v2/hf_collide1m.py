"""Safe helpers for materializing tiny COLLIDE-1M Hugging Face subsets."""

from __future__ import annotations

import json
import math
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SUMMARY_COLUMNS = [
    "FullReco_MET_MET",
    "FullReco_MET_Phi",
    "FullReco_JetAK4_PT",
    "FullReco_Electron_PT",
    "FullReco_MuonTight_PT",
    "L1T_MET_MET",
    "L1T_JetAK4_PT",
]
MAX_ROWS_RAW_MODE = 100
MAX_ROWS_PER_CHUNK = 10


def materialize_hf_collide1m_subset(
    dataset_name: str,
    split: str,
    output_dir: str | Path,
    max_rows: int = 20,
    max_files: int = 1,
    seed: int = 42,
    data_dir: str | None = None,
    data_files: str | list[str] | None = None,
    columns: list[str] | None = None,
    materialization_mode: str = "summary",
) -> dict[str, Any]:
    """Materialize a small streaming HF subset as local parquet.

    Summary mode is the safe default: selected variable-length columns are
    converted into scalar event-level statistics before writing parquet.
    """

    if max_rows < 0:
        raise ValueError("max_rows must be non-negative.")
    if max_files <= 0:
        raise ValueError("max_files must be positive.")
    if materialization_mode not in {"summary", "raw"}:
        raise ValueError("materialization_mode must be either 'summary' or 'raw'.")
    if materialization_mode == "raw" and max_rows > MAX_ROWS_RAW_MODE:
        raise ValueError(
            f"Raw HF materialization is capped at {MAX_ROWS_RAW_MODE} rows. "
            "Use summary mode for safer smoke tests."
        )

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install with: pip install datasets huggingface_hub") from exc

    output_path = Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    summary_columns = columns or DEFAULT_SUMMARY_COLUMNS

    dataset = _load_streaming_dataset(
        load_dataset=load_dataset,
        dataset_name=dataset_name,
        split=split,
        data_dir=data_dir,
        data_files=data_files,
        columns=summary_columns if materialization_mode == "summary" else columns,
    )
    _clear_previous_materialization(output_path)

    rows_written = 0
    parquet_files: list[str] = []
    writer: pq.ParquetWriter | None = None
    current_file_index = 0
    rows_in_current_file = 0
    rows_per_file = max(1, math.ceil(max_rows / max_files)) if max_rows else 1

    try:
        row_iter = iter(dataset)
        while rows_written < max_rows and current_file_index < max_files:
            rows_to_take = min(
                MAX_ROWS_PER_CHUNK,
                max_rows - rows_written,
                rows_per_file - rows_in_current_file,
            )
            if rows_to_take <= 0:
                if writer is not None:
                    writer.close()
                    writer = None
                current_file_index += 1
                rows_in_current_file = 0
                continue

            raw_rows = list(islice(row_iter, rows_to_take))
            if not raw_rows:
                break
            if materialization_mode == "summary":
                rows = [
                    summarize_hf_event_row(
                        row,
                        columns=summary_columns,
                        process_name=data_dir,
                        process_id=_stable_process_id(data_dir),
                        event_id=rows_written + offset,
                    )
                    for offset, row in enumerate(raw_rows)
                ]
            else:
                rows = raw_rows

            table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
            if writer is None:
                parquet_name = f"part-{current_file_index:05d}.parquet"
                parquet_path = output_path / parquet_name
                writer = pq.ParquetWriter(parquet_path, table.schema)
                parquet_files.append(parquet_name)
            writer.write_table(table)
            chunk_rows = len(rows)
            rows_written += chunk_rows
            rows_in_current_file += chunk_rows
    finally:
        if writer is not None:
            writer.close()

    manifest = {
        "dataset_name": dataset_name,
        "source_dataset_name": dataset_name,
        "split": split,
        "seed": seed,
        "data_dir": data_dir,
        "data_files": data_files,
        "max_rows": max_rows,
        "max_files": max_files,
        "materialization_mode": materialization_mode,
        "summary_columns": summary_columns,
        "rows_written": rows_written,
        "parquet_files": parquet_files,
        "warning": "Streaming tiny subset only; never use this helper to download the full COLLIDE-1M dataset.",
    }
    manifest_path = output_path / "hf_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def summarize_hf_event_row(
    row: dict[str, Any],
    columns: list[str],
    process_name: str | None = None,
    process_id: int | None = None,
    event_id: int | None = None,
) -> dict[str, Any]:
    """Convert one HF event row with array-like values into scalar summaries."""

    summary: dict[str, Any] = {"event_id": int(event_id or 0)}
    if process_name is not None:
        summary["process_name"] = process_name
    if process_id is not None:
        summary["process_id"] = int(process_id)

    for column in columns:
        values = _to_numeric_array(row.get(column))
        summary[f"{column}_len"] = int(values.size)
        if values.size == 0:
            summary[f"{column}_mean"] = 0.0
            summary[f"{column}_std"] = 0.0
            summary[f"{column}_min"] = 0.0
            summary[f"{column}_max"] = 0.0
            summary[f"{column}_sum"] = 0.0
            continue
        summary[f"{column}_mean"] = float(values.mean())
        summary[f"{column}_std"] = float(values.std())
        summary[f"{column}_min"] = float(values.min())
        summary[f"{column}_max"] = float(values.max())
        summary[f"{column}_sum"] = float(values.sum())
    return summary


def _load_streaming_dataset(
    load_dataset: Any,
    dataset_name: str,
    split: str,
    data_dir: str | None,
    data_files: str | list[str] | None,
    columns: list[str] | None,
) -> Any:
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if data_dir is not None:
        kwargs["data_dir"] = data_dir
    if data_files is not None:
        kwargs["data_files"] = data_files
    if columns is not None:
        kwargs["columns"] = columns
    try:
        return load_dataset(dataset_name, **kwargs)
    except TypeError:
        kwargs.pop("columns", None)
        return load_dataset(dataset_name, **kwargs)


def _clear_previous_materialization(output_path: Path) -> None:
    for parquet_path in output_path.glob("part-*.parquet"):
        parquet_path.unlink()
    manifest_path = output_path / "hf_subset_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _to_numeric_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if hasattr(value, "to_numpy"):
        value = value.to_numpy()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        array = np.asarray(list(_iter_numeric_values(value)), dtype=np.float64)
    if array.size == 0:
        return array.astype(np.float64)
    return array[np.isfinite(array)]


def _iter_numeric_values(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes)):
        return
    try:
        iterator = iter(value)
    except TypeError:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(number):
            yield number
        return

    for item in iterator:
        yield from _iter_numeric_values(item)


def _stable_process_id(process_name: str | None) -> int | None:
    if process_name is None:
        return None
    # Deterministic compact integer without relying on Python's randomized hash.
    return sum(ord(char) for char in process_name) % 1_000_000


__all__ = ["materialize_hf_collide1m_subset", "summarize_hf_event_row"]
