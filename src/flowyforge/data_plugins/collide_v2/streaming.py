"""Frozen Parquet inventories, uniform samples and bounded COLLIDE event streams.

Legacy full runs store row-group metadata and an implicit seeded permutation.
Opt-in portable partitions store explicit event identities, independent of view.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
import time as clock
import copy
from contextlib import closing
import tempfile
from datetime import date, datetime, time
from decimal import Decimal
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from flowyforge.core.typing import EventBatch
from flowyforge.core.parallel import map_files
from flowyforge.data_plugins.collide_v2.provenance import (
    ORIGIN_COLUMNS, event_id, logical_origin, pop_origin, provenance_columns, sha256_file,
)
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
    result = {"columns": columns, "missing_optional_columns": sorted(set(requested) - set(schema.names)), "units": unit_info}
    if origins := provenance_columns(schema):
        result["provenance_columns"] = origins
    return result


def _split_counts(count: int, validation_fraction: float, test_fraction: float) -> dict[str, int]:
    if count < 3:
        raise ValueError(f"Only {count} selected events; at least 3 per process are required for nonempty splits.")
    validation = min(max(1, int(count * validation_fraction)), count - 2)
    test = min(max(1, int(count * test_fraction)), count - validation - 1)
    return {"train": count - validation - test, "validation": validation, "test": test}


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
        splits = _split_counts(count, validation_fraction, test_fraction)
        manifest["processes"].append({"name": name, "label": label, "files": files, "num_rows": total,
                                      "selected_rows": count, "selected_indices": selected,
                                      "permutation_key": _seed(seed, name + ':split'), "split_counts": splits})
        manifest["label_map"][name] = label
        for split, value in splits.items():
            manifest["split_counts"][split] += value
        print(f"  {name}: {len(files)} files, {total:,} rows, {count:,} selected; {splits}", flush=True)
    return manifest


PARTITION_ALGORITHM = "stable-origin-uniform-cap-feistel6-v1"
SPLITS = ("train", "validation", "test")


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=_json_value).encode()).hexdigest()


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, (date, datetime, time, Decimal)):
        return {type(value).__name__: str(value)}
    raise TypeError(f"Unsupported Parquet value for content fingerprint: {type(value).__name__}")


def _field_signature(field):
    # Ignore Arrow's incidental list child name (item/element), not value types.
    dtype = field.type
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
        kind = "large_list" if pa.types.is_large_list(dtype) else "list"
        signature = [kind, getattr(dtype, "list_size", None), _field_signature(dtype.value_field)[1:]]
    elif pa.types.is_struct(dtype):
        signature = ["struct", sorted((_field_signature(child) for child in dtype), key=lambda x: x[0])]
    else:
        signature = str(dtype)
    return [field.name, signature, field.nullable,
            sorted((key.hex(), value.hex()) for key, value in (field.metadata or {}).items())]


def _schema_digest(schema):
    return _json_digest({
        "fields": sorted((_field_signature(field) for field in schema
                          if field.name not in ORIGIN_COLUMNS), key=lambda x: x[0]),
        "metadata": sorted((key.hex(), value.hex()) for key, value in (schema.metadata or {}).items()),
    })


def partition_policy(manifest):
    """v1 always means strict; a policy flag cannot downgrade an old manifest."""
    return "strict" if manifest["version"] == 1 else manifest["check_integrity_SHA256"]


def _policy(value):
    if value not in ("fast", "strict"):
        raise ValueError("check_integrity_SHA256 must be exactly 'fast' or 'strict'.")
    return value


def _event_records(table, name, *, policy, digest=None, row_start=0, timing=None):
    """Common export/scan records; fast projects origins BEFORE Python conversion."""
    origins = provenance_columns(table.schema)
    signature = _schema_digest(table.schema) if policy == "strict" else None
    rows = table if policy == "strict" else table.select(origins)
    records = []
    for index, row in enumerate(rows.to_pylist()):
        origin = pop_origin(row) or {"source": "local", "revision": None,
            "process": name, "file": "sha256:" + digest, "row": row_start + index, "sha256": digest}
        if origin["process"] != name:
            raise ValueError(f"Origin process {origin['process']!r} differs from assigned process {name!r}.")
        record = {"id": event_id(origin), "origin": logical_origin(origin)}
        if policy == "strict":
            started = clock.perf_counter()
            record["content_sha256"] = _json_digest([signature, row])
            if timing is not None:
                timing["event_fingerprinting_seconds"] += clock.perf_counter() - started
        records.append(record)
    return records


def _scan_partition_file(job):
    name, path, policy, read_batch_size, previous = job
    before = path.stat()
    timing = {"event_fingerprinting_seconds": 0., "file_sha256_seconds": 0.}
    with pq.ParquetFile(path) as parquet:
        origins = provenance_columns(parquet.schema_arrow)
        schema_digest = _schema_digest(parquet.schema_arrow)
        digest, records = None, None
        if policy == "strict" or not origins:
            # Raw local identities need ONE content-addressed original. Later fast
            # bindings may reuse it only with unambiguous matching file metadata.
            candidates = [f for f in previous.get("files", []) if policy == "fast"
                          and f.get("name") == path.name and f["size_bytes"] == before.st_size
                          and f["mtime_ns"] == before.st_mtime_ns and f["schema_sha256"] == schema_digest]
            if len(candidates) == 1:
                digest = candidates[0].get("origin_sha256")
            if digest is None:
                if previous and policy == "fast" and not origins:
                    raise ValueError(f"Fast raw source metadata changed/ambiguous: {path}; strict verification or new partitions required.")
                started = clock.perf_counter()
                digest = sha256_file(path)
                timing["file_sha256_seconds"] = clock.perf_counter() - started
        if policy == "strict":
            known_files = {f["sha256"]: f for f in previous.get("files", [])}
            if digest in known_files:
                known_events = {e["id"]: e for e in previous["events"]}
                records = [dict(known_events[key]) for key in known_files[digest]["event_ids"]]
        if records is None:
            records, row_start = [], 0
            for batch in parquet.iter_batches(batch_size=read_batch_size, use_threads=False,
                                              columns=None if policy == "strict" else origins):
                # Zero-column Arrow batches preserve the row count for raw fast sources.
                records.extend(_event_records(pa.Table.from_batches([batch]), name, policy=policy,
                                              digest=digest, row_start=row_start, timing=timing))
                row_start += len(batch)
        if len(records) != parquet.metadata.num_rows:
            raise ValueError(f"Cached partition row count differs: {path}")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source file changed during partition scan: {path}")
    info = {"size_bytes": before.st_size, "num_rows": len(records), "event_ids": [e["id"] for e in records]}
    if policy == "strict":
        info["sha256"] = digest
    else:
        info.update(name=path.name, mtime_ns=before.st_mtime_ns, schema_sha256=schema_digest)
        if not origins:
            info["origin_sha256"] = digest  # Identity material, NOT a renewed content verification.
    return name, path, info, records, timing


def _scan_partition_source(source, *, read_batch_size=2048, cached=None,
                           check_integrity_SHA256="strict", num_workers=1, performance=None):
    """Strict: all values/file hashes. Fast: footer + provenance columns only."""
    if type(read_batch_size) is not int or read_batch_size < 1:
        raise ValueError("read_batch_size must be a positive integer.")
    policy = _policy(check_integrity_SHA256)
    result, seen = [], set()
    cache = {process["name"]: process for process in (cached or {}).get("processes", [])}
    paths_by_name = _resolve_processes(source)
    grouped = {name: {"name": name, "paths": paths, "files": [], "events": []} for name, paths in paths_by_name.items()}
    jobs = ((name, path, policy, read_batch_size, cache.get(name, {})) for name, paths in paths_by_name.items() for path in paths)
    with closing(map_files(_scan_partition_file, jobs, num_workers)) as results:
        for name, path, info, records, timing in results:
            ids = info["event_ids"]
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError(f"Duplicate event identity in partition source: {path}")
            seen.update(ids)
            grouped[name]["files"].append(info)
            grouped[name]["events"].extend(records)
            if performance is not None:
                for key, value in timing.items():
                    performance[key] = performance.get(key, 0.) + value
            print(f"  Partition inventory {name}: {path.name}, {len(ids):,} rows ({policy})", flush=True)
    for process in grouped.values():
        process["events"].sort(key=lambda event: event["id"])
        result.append(process)
    return result


def create_partition_manifest(source: Mapping[str, Any], *, seed=42, validation_fraction=0.2,
                              test_fraction=0.2, read_batch_size=2048, check_integrity_SHA256="strict",
                              num_workers=1, performance=None, _collected=None) -> dict[str, Any]:
    """Freeze source-row memberships once, without a view or object reader.

    The current COLLIDE reader has no verified physical-ID contract. Identity is
    therefore original logical source/process/file/row (with revision/hash),
    never an inferred run/event column or an absolute cache path.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if any(not math.isfinite(value) or value <= 0 for value in (validation_fraction, test_fraction)) or validation_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must be positive and sum to less than one.")
    cap = source.get("max_events_per_class")
    if cap is not None and (type(cap) is not int or cap < 3):
        raise ValueError("max_events_per_class must be >= 3 or null.")
    policy = _policy(check_integrity_SHA256)
    processes = copy.deepcopy(_collected) if _collected is not None else _scan_partition_source(
        source, read_batch_size=read_batch_size, check_integrity_SHA256=policy, num_workers=num_workers, performance=performance)
    manifest = {"kind": "flowyforge.event_partitions", "version": 1 if policy == "strict" else 2, "algorithm": PARTITION_ALGORITHM,
                "identity": "origin-v1", "physical_deduplication": False, "seed": seed,
                "validation_fraction": validation_fraction, "test_fraction": test_fraction,
                "max_events_per_class": cap, "processes": processes, "label_map": {},
                "split_counts": dict.fromkeys(SPLITS, 0)}
    if policy == "fast":
        manifest["check_integrity_SHA256"] = "fast"
    for label, process in enumerate(processes):
        process.pop("paths", None)  # Physical locations never enter the portable artifact.
        process["files"].sort(key=lambda info: info["sha256"] if policy == "strict" else info["event_ids"])
        process["events"].sort(key=lambda event: event["id"])
        name, events = process["name"], process["events"]
        total = len(events)
        count = min(total, cap) if cap is not None else total
        counts = _split_counts(count, validation_fraction, test_fraction)
        selected = sorted(random.Random(_seed(seed, name + ':sample')).sample(range(total), count)) if count < total else range(total)
        ranks = _permuted_ranks(np.arange(count), count, _seed(seed, name + ':split'))
        for event in events:
            event["split"] = None
        for index, rank in zip(selected, ranks):
            events[index]["split"] = ("train" if rank < counts["train"] else
                                      "validation" if rank < counts["train"] + counts["validation"] else "test")
        process.update(label=label, num_rows=total, selected_rows=count, split_counts=counts)
        manifest["label_map"][name] = label
        for split in SPLITS:
            manifest["split_counts"][split] += counts[split]
    manifest["sha256"] = _json_digest(manifest)
    validate_partition_manifest(manifest)
    return manifest


def validate_partition_manifest(manifest):
    """Reject unknown versions, damaged identities, counts or memberships."""
    if (not isinstance(manifest, dict) or manifest.get("kind") != "flowyforge.event_partitions"
        or manifest.get("version") not in (1, 2) or manifest.get("algorithm") != PARTITION_ALGORITHM):
        raise ValueError("Unsupported partition manifest kind/version/algorithm; supported: strict v1, fast v2.")
    if manifest.get("sha256") != _json_digest({key: value for key, value in manifest.items() if key != "sha256"}):
        raise ValueError("Partition manifest checksum mismatch; do not edit saved memberships.")
    try:
        policy = partition_policy(manifest)
        if ((manifest["version"] == 1 and manifest.get("check_integrity_SHA256", "strict") != "strict")
            or (manifest["version"] == 2 and policy != "fast")):
            raise ValueError("Partition version/integrity policy mismatch; strict v1 cannot be downgraded.")
        seen, totals, labels = set(), dict.fromkeys(SPLITS, 0), {}
        if manifest["identity"] != "origin-v1" or manifest["physical_deduplication"] is not False:
            raise ValueError("Unsupported partition identity policy.")
        if type(manifest["seed"]) is not int or manifest["seed"] < 0:
            raise ValueError("Invalid partition seed.")
        vf, tf, cap = manifest["validation_fraction"], manifest["test_fraction"], manifest["max_events_per_class"]
        if any(not math.isfinite(x) or x <= 0 for x in (vf, tf)) or vf + tf >= 1:
            raise ValueError("Invalid partition fractions.")
        if cap is not None and (type(cap) is not int or cap < 3):
            raise ValueError("Invalid partition sample cap.")
        for label, process in enumerate(manifest["processes"]):
            ids, counts = set(), dict.fromkeys(SPLITS, 0)
            if process["label"] != label or process["name"] in labels:
                raise ValueError("Invalid partition label mapping.")
            labels[process["name"]] = label
            for event in process["events"]:
                key = event["id"]
                if key in seen or key != event_id(event["origin"]) or event["origin"]["process"] != process["name"]:
                    raise ValueError("Duplicate or invalid partition event identity.")
                if policy == "strict" and (len(event["content_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in event["content_sha256"])):
                    raise ValueError("Invalid partition event fingerprint.")
                if policy == "fast" and "content_sha256" in event:
                    raise ValueError("Fast partitions must not claim per-event content verification.")
                seen.add(key)
                ids.add(key)
                if event["split"] is not None:
                    counts[event["split"]] += 1
            selected = min(len(ids), cap) if cap is not None else len(ids)
            if (process["num_rows"] != len(ids) or process["selected_rows"] != selected or
                counts != process["split_counts"] or counts != _split_counts(selected, vf, tf)):
                raise ValueError("Inconsistent partition split counts.")
            file_ids = []
            for info in process["files"]:
                fingerprint = info["sha256"] if policy == "strict" else info["schema_sha256"]
                if (len(info["event_ids"]) != info["num_rows"] or info["size_bytes"] < 0 or
                    len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)):
                    raise ValueError("Invalid partition file inventory.")
                if policy == "fast":
                    if (type(info["mtime_ns"]) is not int or info["mtime_ns"] < 0 or not info["name"]
                        or Path(info["name"]).name != info["name"] or "sha256" in info):
                        raise ValueError("Invalid fast file metadata (no full-file verification claimed).")
                    if "origin_sha256" in info and (len(info["origin_sha256"]) != 64 or
                            any(c not in "0123456789abcdef" for c in info["origin_sha256"])):
                        raise ValueError("Invalid original identity checksum.")
                file_ids.extend(info["event_ids"])
            if len(file_ids) != len(ids) or set(file_ids) != ids:
                raise ValueError("Partition file inventory differs from event identities.")
            for split in SPLITS:
                totals[split] += counts[split]
        if len(labels) < 2 or labels != manifest["label_map"] or totals != manifest["split_counts"]:
            raise ValueError("Invalid partition process counts/labels.")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"Malformed partition manifest: {exc}") from exc


def save_partition_manifest(manifest, path):
    """Exclusive creation: partitions are immutable experiment artifacts."""
    validate_partition_manifest(manifest)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Publish a fully written artifact atomically, while refusing concurrent overwrite.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_partition_manifest(path):
    manifest = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    validate_partition_manifest(manifest)
    return manifest


def bind_partition_manifest(partitions, source, *, read_batch_size=2048, **reader_settings):
    """Resolve immutable memberships onto current files and one object view.

    Re-exports must contain the same complete event universe and raw payloads.
    No RNG or sampling runs here. A binding is disposable; partitions are not.
    """
    validate_partition_manifest(partitions)
    cap = source.get("max_events_per_class")
    if cap is not None and cap != partitions["max_events_per_class"]:
        raise ValueError("Sample cap differs from partition manifest; remove the override or use its original cap.")
    policy = partition_policy(partitions)
    current = _scan_partition_source(source, read_batch_size=read_batch_size, cached=partitions,
                                     check_integrity_SHA256=policy)
    if [p["name"] for p in current] != [p["name"] for p in partitions["processes"]]:
        raise ValueError("Partition source processes changed.")
    reader = _reader_config(**reader_settings)
    binding = {"version": 2, "algorithm": "partition-binding-v1", "seed": partitions["seed"],
               "partition_sha256": partitions["sha256"], "reader": reader,
               "validation_fraction": partitions["validation_fraction"], "test_fraction": partitions["test_fraction"],
               "label_map": dict(partitions["label_map"]), "split_counts": dict(partitions["split_counts"]), "processes": []}
    if policy == "fast":
        binding["check_integrity_SHA256"] = "fast"
    task_id = 0
    for stored, actual in zip(partitions["processes"], current):
        expected = {event["id"]: event for event in stored["events"]}
        found = {event["id"]: event for event in actual["events"]}
        if expected.keys() != found.keys():
            raise ValueError(f"Partition source identities changed for {stored['name']}: missing/extra original rows.")
        if policy == "strict" and any(event["content_sha256"] != found[key]["content_sha256"] for key, event in expected.items()):
            raise ValueError(f"Partition source content/schema changed for {stored['name']}.")
        if policy == "fast":
            inventories = {tuple(f["event_ids"]): f for f in stored["files"]}
            for info in actual["files"]:
                old = inventories.get(tuple(info["event_ids"]))
                if old is None or any(info[k] != old[k] for k in ("size_bytes", "mtime_ns", "schema_sha256", "num_rows")):
                    raise ValueError(f"Fast partition file metadata/schema changed for {stored['name']}; "
                                     "preserve files/timestamps on relocation or create new partitions. No content equivalence inferred.")
        files, indices, total = [], {split: [] for split in SPLITS}, 0
        for path, inventory in zip(actual["paths"], actual["files"]):
            stat = path.stat()
            if policy == "fast" and (stat.st_size != inventory["size_bytes"] or stat.st_mtime_ns != inventory["mtime_ns"]):
                raise ValueError(f"Source file changed during fast partition binding: {path}")
            with pq.ParquetFile(path) as parquet:
                info = _schema_info(parquet.schema_arrow, reader, path)
                groups = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
                files.append({**inventory, **info, "path": str(path), "offset": total,
                              "mtime_ns": stat.st_mtime_ns, "row_groups": groups, "task_offset": task_id})
                if policy == "fast":
                    files[-1]["check_integrity_SHA256"] = "fast"
                task_id += len(groups)
            for row_index, key in enumerate(inventory["event_ids"]):
                split = expected[key]["split"]
                if split is not None:
                    indices[split].append(total + row_index)
            total += inventory["num_rows"]
        binding["processes"].append({"name": stored["name"], "label": stored["label"], "files": files,
            "num_rows": total, "selected_rows": stored["selected_rows"], "split_counts": dict(stored["split_counts"]),
            "selected_indices": sorted(index for values in indices.values() for index in values), "split_indices": indices})
    return binding


def _check_file(info: dict[str, Any]) -> None:
    stat = Path(info["path"]).stat()
    if stat.st_size != info["size_bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
        raise ValueError(f"Source file changed since manifest creation: {info['path']}")
    if "sha256" in info and sha256_file(info["path"]) != info["sha256"]:
        raise ValueError(f"Source file content changed since partition binding: {info['path']}")
    if info.get("check_integrity_SHA256") == "fast":
        with pq.ParquetFile(info["path"]) as parquet:
            provenance_columns(parquet.schema_arrow)
            if (parquet.metadata.num_rows != info["num_rows"] or
                _schema_digest(parquet.schema_arrow) != info["schema_sha256"]):
                raise ValueError(f"Fast source schema/row count changed: {info['path']}")


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
    explicit_split = split is not None and "split_indices" in process
    if explicit_split:
        selected = process["split_indices"][split]
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
                                               columns=list(info["columns"]) + info.get("provenance_columns", []), use_threads=False):
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
                if split is not None and not explicit_split and len(indices):
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
                            origin = pop_origin(row)
                            event = _convert_row(row, reader, specs, process["label"])
                        except ValueError as exc:
                            raise ValueError(f"{info['path']}, row {int(index) - info['offset']}: {exc}") from exc
                        event.metadata.update(source_file=info["path"], source_row=int(index) - info["offset"], process_name=process["name"])
                        if origin is None and "event_ids" in info:
                            original_digest = info.get("origin_sha256", info.get("sha256"))
                            origin = {"source": "local", "revision": None, "process": process["name"],
                                      "file": "sha256:" + original_digest, "sha256": original_digest,
                                      "row": int(index) - info["offset"]}
                        if origin is not None:
                            event.metadata["origin"] = origin
                            event.metadata["event_id"] = event_id(origin)
                            if "event_ids" in info and event.metadata["event_id"] != info["event_ids"][int(index) - info["offset"]]:
                                raise ValueError(f"Event identity differs from partition binding: {info['path']}")
                        if audit:
                            event.metadata["raw_stats"] = _raw_stats(row)
                        yield event
                local += len(batch)


class StreamingDataset(IterableDataset):
    """Worker-sharded row groups, proportionally mixed classes and bounded shuffle."""

    def __init__(self, manifest, split, *, read_batch_size=128, shuffle_buffer=512,
                 purpose="downstream", **reader_overrides):
        super().__init__()
        if (manifest.get("version"), manifest.get("algorithm")) not in {
            (1, "uniform-cap-feistel6-split-v1"), (2, "partition-binding-v1")
        }:
            raise ValueError("Unsupported dataset manifest version/algorithm.")
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation or test.")
        if purpose not in {"downstream", "pretraining", "checkpoint_selection"}:
            raise ValueError("purpose must be downstream, pretraining or checkpoint_selection.")
        if purpose == "pretraining" and (split != "train" or manifest["version"] != 2):
            raise ValueError("Pretraining requires shared partitions (dataset manifest v2) and the train split only.")
        if purpose == "checkpoint_selection" and split != "validation":
            raise ValueError("Checkpoint selection requires the validation split only.")
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
    result["warnings"] = _audit_warnings(result)
    return result


def _audit_warnings(result):
    """Warnings from final totals (also used after merging per-file audits)."""
    warnings = []
    if result["unit_assumed_fields"]:
        warnings.append(f"Units are configuration assumptions for {result['unit_assumed_fields']} file/field entries without recognized metadata; inspect raw ranges and confirm the production convention.")
    for name, totals in result["collections"].items():
        if totals["truncated_events"] or totals["nonfinite_values_replaced"] or totals["missing_optional_events"]:
            warnings.append(f"{name} audit: {totals['truncated_events']} truncated events, {totals['nonfinite_values_replaced']} nonfinite values replaced, {totals['missing_optional_events']} events with missing optional fields.")
    return warnings


__all__ = ["build_manifest", "inspect_dataset", "StreamingDataset", "create_partition_manifest",
           "bind_partition_manifest", "save_partition_manifest", "load_partition_manifest",
           "validate_partition_manifest"]
