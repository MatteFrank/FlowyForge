"""Frozen Parquet inventories, uniform samples and bounded COLLIDE event streams.

Full runs store file/row-group metadata, never an index for every event. A seeded
permutation assigns exact, disjoint per-class splits without materializing it.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from flowyforge.core.typing import EventBatch
from flowyforge.data_plugins.collide_v2.object_dataset import (
    _object_values, default_collection_specs, expand_parquet_paths, required_columns, row_to_event_batch,
)


def _seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest(), "little")


def _permuted_ranks(indices: np.ndarray, size: int, key: int) -> np.ndarray:
    """Six-round Feistel permutation, cycle-walked onto [0, size).

    SplitMix64 round mixing is deterministic, nonlinear and inexpensive; this
    is a sampling permutation, not a cryptographic primitive. Work and memory
    are bounded by the current Arrow batch (domain acceptance is at least 1/4).
    """
    if size <= 1:
        return np.zeros(len(indices), dtype=np.int64)
    bits = max(1, ((size - 1).bit_length() + 1) // 2)
    mask = np.uint64((1 << bits) - 1)
    result = np.asarray(indices, dtype=np.uint64).copy()
    pending = np.arange(len(result))
    keys = [np.uint64((key + round_id * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)) for round_id in range(6)]
    with np.errstate(over="ignore"):
        while len(pending):
            left, right = result[pending] >> np.uint64(bits), result[pending] & mask
            for round_key in keys:
                mixed = right + round_key
                mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                mixed ^= mixed >> np.uint64(31)
                left, right = right, left ^ (mixed & mask)
            result[pending] = (left << np.uint64(bits)) | right
            pending = pending[result[pending] >= size]
    return result.astype(np.int64)


def _walk_parquet(root: Path) -> list[Path]:
    def fail(error: OSError) -> None:
        raise error
    return sorted(path.resolve() for directory, _, files in os.walk(root, onerror=fail)
                  for name in files if name.endswith(".parquet")
                  if (path := Path(directory) / name).is_file())


def _resolve_processes(source: Mapping[str, Any]) -> dict[str, list[Path]]:
    explicit = source.get("processes")
    if explicit:
        if not isinstance(explicit, Mapping):
            raise ValueError("processes must map class names to Parquet paths.")
        entries = dict(explicit)
    else:
        root = Path(source.get("root", "")).expanduser()
        if not source.get("root"):
            raise ValueError("Provide processes or a dataset root.")
        children = sorted(root.iterdir())  # Preserve permission errors; never silently skip EOS.
        if any(path.is_file() and path.suffix == ".parquet" for path in children):
            raise ValueError("Unlabeled Parquet files at dataset root: use explicit --process NAME=PATH mappings.")
        directories = [path for path in children if path.is_dir() and not path.name.startswith(".")]
        wrappers = [path for path in directories if path.name.startswith("production")]
        if wrappers:
            if len(directories) != 1 or wrappers[0].name != "production_final":
                raise ValueError("Ambiguous production directories: set --data-root to one production or use --process mappings.")
            root = wrappers[0]
            directories = sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))
        entries = {path.name: path for path in directories}
    selected_names = source.get("process_names")
    if selected_names is not None:
        if not isinstance(selected_names, list) or not selected_names or set(selected_names) - entries.keys():
            raise ValueError(f"process_names must select existing process directories: {sorted(entries)}")
        entries = {name: entries[name] for name in selected_names}
    if len(entries) < 2 or any(not isinstance(name, str) or not name.strip() for name in entries):
        raise ValueError("Classification needs at least two named processes.")
    result, used = {}, set()
    for name, paths in sorted(entries.items()):
        paths = [paths] if isinstance(paths, (str, Path)) else paths
        files = set()
        for entry in paths:
            path = Path(entry).expanduser()
            files.update(_walk_parquet(path) if path.is_dir() else expand_parquet_paths(entry))
        if not files:
            raise ValueError(f"No Parquet files for process {name}; select process directories explicitly.")
        if overlap := used.intersection(files):
            raise ValueError(f"Parquet files assigned to multiple processes: {sorted(map(str, overlap))}")
        used.update(files)
        result[name] = sorted(files)
    return result


def _reader_config(view="FullReco", collection_limits=None, units=None, eta_clip=5.0, value_clip=1e6):
    units = dict(units or {"momentum": "GeV", "angle": "rad"})
    if units.get("momentum") not in {"GeV", "MeV"} or units.get("angle") not in {"rad", "deg"}:
        raise ValueError("units require momentum: GeV|MeV and angle: rad|deg.")
    specs = default_collection_specs(collection_limits)
    required_columns(view, specs)
    if any(spec.max_objects is None for spec in specs):
        raise ValueError("Streaming requires finite collection_limits for every enabled collection.")
    if not all(math.isfinite(value) and value > 0 for value in (eta_clip, value_clip)):
        raise ValueError("eta_clip and value_clip must be finite and positive.")
    return {"view": view, "collection_limits": dict(collection_limits), "units": units,
            "eta_clip": eta_clip, "value_clip": value_clip}


def _numeric_list(dtype: pa.DataType) -> bool:
    depth = 0
    while pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        depth += 1
        dtype = dtype.value_type
    return depth > 0 and (pa.types.is_integer(dtype) or pa.types.is_floating(dtype)
                          or pa.types.is_decimal(dtype) or pa.types.is_null(dtype))


def _schema_info(schema: pa.Schema, reader: dict[str, Any], path: Path) -> dict[str, Any]:
    specs = default_collection_specs(reader["collection_limits"])
    required = set(required_columns(reader["view"], specs))
    if missing := required - set(schema.names):
        raise ValueError(f"{path}: missing required nested COLLIDE columns: {sorted(missing)}")
    requested = required_columns(reader["view"], specs, include_optional=True)
    columns, unit_info = {}, {}
    metadata = schema.metadata or {}
    try:
        unit_map = json.loads(metadata.get(b"units", b"{}"))
    except (ValueError, UnicodeDecodeError):
        unit_map = {}
    if not isinstance(unit_map, dict):
        unit_map = {}
    aliases = {"gev": "GeV", "mev": "MeV", "rad": "rad", "radian": "rad", "radians": "rad",
               "deg": "deg", "degree": "deg", "degrees": "deg"}
    for column in requested:
        if column not in schema.names:
            continue
        field = schema.field(column)
        if not _numeric_list(field.type):
            raise ValueError(f"{path}: {column} must be a numeric object list, got {field.type}.")
        columns[column] = str(field.type)
        quantity = column.rsplit("_", 1)[-1]
        kind = "momentum" if quantity in {"PT", "MET", "Mass"} else "angle" if quantity == "Phi" else None
        if kind:
            field_metadata = field.metadata or {}
            declared = field_metadata.get(b"unit", field_metadata.get(b"units", unit_map.get(column)))
            if declared is None:
                declared = metadata.get(f"{kind}_unit".encode())
            if isinstance(declared, bytes):
                declared = declared.decode("utf-8")
            effective = reader["units"][kind]
            if declared is not None and aliases.get(str(declared).strip().lower()) != effective:
                raise ValueError(f"{path}: unit metadata for {column} is {declared!r}, configured {effective!r}.")
            unit_info[column] = {"unit": effective, "status": "metadata" if declared is not None else "assumed"}
    return {"columns": columns, "missing_optional_columns": sorted(set(requested) - set(schema.names)), "units": unit_info}


def build_manifest(source: Mapping[str, Any], *, seed=42, validation_fraction=0.2,
                   test_fraction=0.2, view="FullReco", collection_limits=None,
                   units=None, eta_clip=5.0, value_clip=1e6) -> dict[str, Any]:
    """Check every footer and freeze the source, sampling and exact split counts."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if any(not math.isfinite(value) or value <= 0 for value in (validation_fraction, test_fraction)) or validation_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    cap = source.get("max_events_per_class")
    if cap is not None and (type(cap) is not int or cap < 3):
        raise ValueError("max_events_per_class must be >= 3 or null.")
    reader = _reader_config(view, collection_limits, units, eta_clip, value_clip)
    manifest = {"version": 1, "algorithm": "uniform-cap-feistel6-split-v1", "seed": seed, "reader": reader,
                "validation_fraction": validation_fraction, "test_fraction": test_fraction,
                "processes": [], "label_map": {}, "split_counts": {"train": 0, "validation": 0, "test": 0}}
    task_id = 0
    for label, (name, paths) in enumerate(_resolve_processes(source).items()):
        files, total = [], 0
        for path in paths:
            stat = path.stat()
            with pq.ParquetFile(path) as parquet:
                info = _schema_info(parquet.schema_arrow, reader, path)
                sizes = [parquet.metadata.row_group(index).num_rows for index in range(parquet.num_row_groups)]
                files.append({"path": str(path), "num_rows": parquet.metadata.num_rows, "offset": total,
                              "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                              "row_groups": sizes, "task_offset": task_id, **info})
                total += parquet.metadata.num_rows
                task_id += len(sizes)
        count = min(total, cap) if cap is not None else total
        if count < 3:
            raise ValueError(f"Process {name} has {count} selected events; at least 3 are required.")
        selected = sorted(random.Random(_seed(seed, name + ':sample')).sample(range(total), count)) if count < total else None
        validation = min(max(1, int(count * validation_fraction)), count - 2)
        test = min(max(1, int(count * test_fraction)), count - validation - 1)
        splits = {"train": count - validation - test, "validation": validation, "test": test}
        manifest["processes"].append({"name": name, "label": label, "files": files, "num_rows": total,
                                      "selected_rows": count, "selected_indices": selected,
                                      "permutation_key": _seed(seed, name + ':split'), "split_counts": splits})
        manifest["label_map"][name] = label
        for split, value in splits.items():
            manifest["split_counts"][split] += value
        print(f"  {name}: {len(files)} files, {total:,} rows, {count:,} selected; {splits}", flush=True)
    return manifest


def _check_file(info: dict[str, Any]) -> None:
    stat = Path(info["path"]).stat()
    if stat.st_size != info["size_bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
        raise ValueError(f"Source file changed since manifest creation: {info['path']}")


def _raw_stats(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for column, value in row.items():
        values = _object_values(value, column)
        finite = values[np.isfinite(values)]
        result[column] = {"null_lists": int(value is None), "values": len(values),
                          "nonfinite_values": int(len(values) - len(finite)),
                          "min": float(finite.min()) if len(finite) else None,
                          "max": float(finite.max()) if len(finite) else None}
    return result


def _convert_row(row, reader, specs, target):
    normalized = dict(row)
    for column, value in row.items():
        quantity = column.rsplit("_", 1)[-1]
        factor = (0.001 if reader["units"]["momentum"] == "MeV" else 1.0) if quantity in {"PT", "MET", "Mass"} else (
            math.pi / 180 if quantity == "Phi" and reader["units"]["angle"] == "deg" else 1.0)
        if value is not None and factor != 1:
            normalized[column] = _object_values(value, column) * factor
    return row_to_event_batch(normalized, reader["view"], specs, target=target,
                              eta_clip=reader["eta_clip"], value_clip=reader["value_clip"])


def _tasks(process, worker_id=0, num_workers=1):
    tasks = []
    for info in process["files"]:
        offset = 0
        for index, size in enumerate(info["row_groups"]):
            if (info["task_offset"] + index) % num_workers == worker_id and size:
                tasks.append((info, index, offset, size))
            offset += size
    return tasks


def _iter_process(process, tasks, split, reader, read_batch_size, *, audit=False, selected_override=None):
    specs = default_collection_specs(reader["collection_limits"])
    selected = process["selected_indices"] if selected_override is None else selected_override
    checked = set()
    for info, group, offset, size in tasks:
        start = info["offset"] + offset
        if selected is not None and bisect.bisect_left(selected, start) == bisect.bisect_left(selected, start + size):
            continue
        if info["path"] not in checked:
            _check_file(info)
            checked.add(info["path"])
        with pq.ParquetFile(info["path"]) as parquet:
            local = offset
            for batch in parquet.iter_batches(batch_size=read_batch_size, row_groups=[group],
                                               columns=list(info["columns"]), use_threads=False):
                global_start = info["offset"] + local
                if selected is None:
                    indices = np.arange(global_start, global_start + len(batch), dtype=np.int64)
                    positions = np.arange(len(batch))
                    sample_positions = indices
                else:
                    lower, upper = bisect.bisect_left(selected, global_start), bisect.bisect_left(selected, global_start + len(batch))
                    indices = np.asarray(selected[lower:upper], dtype=np.int64)
                    positions = indices - global_start
                    sample_positions = np.arange(lower, upper, dtype=np.int64)
                if split is not None and len(indices):
                    ranks = _permuted_ranks(sample_positions, process["selected_rows"], process["permutation_key"])
                    train_end = process["split_counts"]["train"]
                    validation_end = train_end + process["split_counts"]["validation"]
                    keep = ranks < train_end if split == "train" else (
                        (ranks >= train_end) & (ranks < validation_end) if split == "validation" else ranks >= validation_end)
                    positions, indices = positions[keep], indices[keep]
                if len(positions):
                    rows = batch.take(pa.array(positions)).to_pylist()
                    for index, row in zip(indices, rows):
                        try:
                            event = _convert_row(row, reader, specs, process["label"])
                        except ValueError as exc:
                            raise ValueError(f"{info['path']}, row {int(index) - info['offset']}: {exc}") from exc
                        event.metadata.update(source_file=info["path"], source_row=int(index) - info["offset"], process_name=process["name"])
                        if audit:
                            event.metadata["raw_stats"] = _raw_stats(row)
                        yield event
                local += len(batch)


class StreamingDataset(IterableDataset):
    """Worker-sharded row groups, proportionally mixed classes and bounded shuffle."""

    def __init__(self, manifest, split, *, read_batch_size=128, shuffle_buffer=512, **reader_overrides):
        super().__init__()
        if manifest.get("version") != 1 or manifest.get("algorithm") != "uniform-cap-feistel6-split-v1":
            raise ValueError("Unsupported dataset manifest version/algorithm.")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test.")
        if any(type(value) is not int or value < 1 for value in (read_batch_size, shuffle_buffer)):
            raise ValueError("read_batch_size and shuffle_buffer must be positive integers.")
        if any(value != manifest["reader"].get(key) for key, value in reader_overrides.items()):
            raise ValueError("Reader overrides differ from the frozen manifest.")
        self.manifest, self.split, self.reader = manifest, split, manifest["reader"]
        self.read_batch_size, self.shuffle_buffer, self.epoch = read_batch_size, shuffle_buffer, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[EventBatch]:
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rng = random.Random(_seed(self.manifest["seed"], f"{self.split}:{self.epoch if self.split == 'train' else 0}:{worker_id}"))
        streams, remaining = [], []
        for process in self.manifest["processes"]:
            tasks = _tasks(process, worker_id, workers)
            if self.split == "train":
                rng.shuffle(tasks)
            if not tasks:
                continue
            streams.append(iter(_iter_process(process, tasks, self.split, self.reader, self.read_batch_size)))
            # Exact with one worker, a scheduling estimate with row-group workers.
            remaining.append(max(1, process["split_counts"][self.split] * sum(task[3] for task in tasks) / process["num_rows"]))
        buffer = []
        try:
            while streams:
                index = rng.choices(range(len(streams)), weights=remaining, k=1)[0]
                try:
                    event = next(streams[index])
                except StopIteration:
                    streams.pop(index)
                    remaining.pop(index)
                    continue
                remaining[index] = max(1, remaining[index] - 1)
                if self.split != "train":
                    yield event
                elif len(buffer) < self.shuffle_buffer:
                    buffer.append(event)
                else:
                    slot = rng.randrange(len(buffer))
                    yield buffer[slot]
                    buffer[slot] = event
            rng.shuffle(buffer)
            yield from buffer
        finally:
            for stream in streams:
                stream.close()


def inspect_dataset(manifest, *, audit_events_per_class=32, read_batch_size=128, **reader_overrides):
    """Uniform bounded raw-value audit; all file schemas were checked at inventory."""
    if any(type(value) is not int or value < 1 for value in (audit_events_per_class, read_batch_size)):
        raise ValueError("Audit and read batch sizes must be positive integers.")
    if any(value != manifest["reader"].get(key) for key, value in reader_overrides.items()):
        raise ValueError("Reader overrides differ from the frozen manifest.")
    result = {"scope": "all file footers; uniform raw-row audit per process (not an exhaustive value audit)",
              "input_units": manifest["reader"]["units"], "model_units": {"momentum": "GeV", "angle": "rad"},
              "num_files": sum(len(process["files"]) for process in manifest["processes"]),
              "split_counts": manifest["split_counts"], "num_events": 0, "processes": {}, "collections": {},
              "fields": {}, "unit_metadata_fields": 0, "unit_assumed_fields": 0, "warnings": []}
    for process in manifest["processes"]:
        for info in process["files"]:
            for units in info["units"].values():
                result["unit_metadata_fields" if units["status"] == "metadata" else "unit_assumed_fields"] += 1
        count = min(audit_events_per_class, process["num_rows"])
        indices = sorted(random.Random(_seed(manifest["seed"], process["name"] + ':audit')).sample(range(process["num_rows"]), count))
        result["processes"][process["name"]] = {"available_rows": process["num_rows"], "selected_rows": process["selected_rows"], "audit_rows": count}
        for event in _iter_process(process, _tasks(process), None, manifest["reader"], read_batch_size,
                                   audit=True, selected_override=indices):
            result["num_events"] += 1
            for name, group in event.objects.items():
                totals = result["collections"].setdefault(name, {"original_objects_total": 0, "retained_objects_total": 0,
                        "truncated_objects_total": 0, "nonfinite_values_replaced": 0, "truncated_events": 0,
                        "missing_optional_events": 0, "missing_optional_fields": []})
                for key, destination in (("original_count", "original_objects_total"), ("retained_count", "retained_objects_total"),
                                         ("truncated_count", "truncated_objects_total"), ("nonfinite_values", "nonfinite_values_replaced")):
                    totals[destination] += group.metadata[key]
                totals["truncated_events"] += int(group.metadata["truncated_count"] > 0)
                totals["missing_optional_events"] += int(bool(group.metadata["missing_optional_fields"]))
                totals["missing_optional_fields"] = sorted(set(totals["missing_optional_fields"]) | set(group.metadata["missing_optional_fields"]))
            for column, stats in event.metadata["raw_stats"].items():
                total = result["fields"].setdefault(column, {"null_lists": 0, "values": 0, "nonfinite_values": 0, "min": None, "max": None})
                for key in ("null_lists", "values", "nonfinite_values"):
                    total[key] += stats[key]
                for key, fn in (("min", min), ("max", max)):
                    if stats[key] is not None:
                        total[key] = stats[key] if total[key] is None else fn(total[key], stats[key])
    if result["unit_assumed_fields"]:
        result["warnings"].append(f"Units are configuration assumptions for {result['unit_assumed_fields']} file/field entries without recognized metadata; inspect raw ranges and confirm the production convention.")
    for name, totals in result["collections"].items():
        if totals["truncated_events"] or totals["nonfinite_values_replaced"] or totals["missing_optional_events"]:
            result["warnings"].append(f"{name} audit: {totals['truncated_events']} truncated events, {totals['nonfinite_values_replaced']} nonfinite values replaced, {totals['missing_optional_events']} events with missing optional fields.")
    return result


__all__ = ["build_manifest", "inspect_dataset", "StreamingDataset"]
