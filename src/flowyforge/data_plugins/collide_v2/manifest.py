"""Manifest generation for COLLIDE-2V parquet datasets."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from flowyforge.data_plugins.collide_v2.eos_paths import list_parquet_files


def scan_parquet_event_counts(dataset_dir: str | Path) -> dict[str, Any]:
    """Count parquet rows using file metadata where possible."""

    root = Path(os.path.expandvars(os.fspath(dataset_dir))).expanduser()
    parquet_files = list_parquet_files(root)
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in parquet_files:
        relative_path = _relative_path(path, root)
        try:
            metadata = pq.ParquetFile(str(path)).metadata
            num_rows = int(metadata.num_rows)
            entries.append(
                {
                    "path": relative_path,
                    "num_rows": num_rows,
                    "process": _infer_process_name(relative_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep bad files visible in manifest.
            errors.append(
                {
                    "path": relative_path,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )

    if not entries and errors:
        raise RuntimeError("No readable parquet files found while scanning event counts.")

    return {
        "dataset_dir": str(root),
        "total_files": len(entries),
        "total_rows": sum(int(entry["num_rows"]) for entry in entries),
        "files": entries,
        "errors": errors,
    }


def write_file_event_counts(counts: dict[str, Any], output_path: str | Path) -> None:
    _write_json(counts, output_path)


def create_split_manifest(
    counts: dict[str, Any],
    split_counts: dict[str, Any] | None,
    classnames: list[str] | None,
    folder_map: dict[str, str] | None,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a simple deterministic split manifest from file event counts."""

    files = list(counts.get("files", []))
    inferred_processes = sorted({str(item.get("process", "unknown")) for item in files})
    resolved_classnames = classnames or inferred_processes
    resolved_folder_map = folder_map or {name: name for name in resolved_classnames}

    grouped_files: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in files:
        grouped_files[str(item.get("process", "unknown"))].append(item)

    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for process, process_files in sorted(grouped_files.items()):
        shuffled = list(process_files)
        rng.shuffle(shuffled)
        process_splits = _split_files(shuffled, _split_spec_for_process(split_counts, process))
        for split_name, split_files in process_splits.items():
            splits.setdefault(split_name, []).extend(
                {
                    "path": item["path"],
                    "process": process,
                    "num_rows": item["num_rows"],
                }
                for item in split_files
            )

    return {
        "seed": seed,
        "dataset_dir": counts.get("dataset_dir"),
        "classnames": resolved_classnames,
        "folder_map": resolved_folder_map,
        "split_counts": split_counts or {},
        "splits": splits,
        "total_files": counts.get("total_files", len(files)),
        "total_rows": counts.get("total_rows", 0),
    }


def write_split_manifest(manifest: dict[str, Any], output_path: str | Path) -> None:
    _write_json(manifest, output_path)


def _split_files(
    files: list[dict[str, Any]],
    split_spec: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    if not split_spec:
        return {"train": files, "val": [], "test": []}

    split_names = list(split_spec)
    if all(isinstance(value, float) and 0 <= value <= 1 for value in split_spec.values()):
        return _split_by_fraction(files, split_spec)

    result: dict[str, list[dict[str, Any]]] = {name: [] for name in split_names}
    cursor = 0
    for split_name, count in split_spec.items():
        take = max(0, int(count))
        result[split_name] = files[cursor : cursor + take]
        cursor += take
    if cursor < len(files):
        result.setdefault("train", []).extend(files[cursor:])
    result.setdefault("train", [])
    result.setdefault("val", [])
    result.setdefault("test", [])
    return result


def _split_by_fraction(
    files: list[dict[str, Any]],
    split_spec: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in split_spec}
    cursor = 0
    total = len(files)
    split_items = list(split_spec.items())
    for index, (split_name, fraction) in enumerate(split_items):
        if index == len(split_items) - 1:
            result[split_name] = files[cursor:]
            break
        take = int(total * float(fraction))
        result[split_name] = files[cursor : cursor + take]
        cursor += take
    result.setdefault("train", [])
    result.setdefault("val", [])
    result.setdefault("test", [])
    return result


def _split_spec_for_process(
    split_counts: dict[str, Any] | None,
    process: str,
) -> dict[str, Any] | None:
    if not split_counts:
        return None
    if process in split_counts and isinstance(split_counts[process], dict):
        return split_counts[process]
    if all(isinstance(value, (int, float)) for value in split_counts.values()):
        return split_counts
    return None


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _infer_process_name(relative_path: str) -> str:
    parent = Path(relative_path).parent
    if parent == Path("."):
        return "unknown"
    return parent.parts[0]


def _write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "create_split_manifest",
    "scan_parquet_event_counts",
    "write_file_event_counts",
    "write_split_manifest",
]
