"""Persistent row origins carried as ordinary Parquet columns, not features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa


PREFIX = "__flowyforge_origin_"
ORIGIN_FIELDS = {
    "source": pa.string(), "revision": pa.string(), "file": pa.string(),
    "row": pa.int64(), "process": pa.string(), "sha256": pa.string(),
}
ORIGIN_COLUMNS = [PREFIX + name for name in ORIGIN_FIELDS]


def logical_origin(origin: dict) -> dict:
    """Normalize origins without using a machine's cache/mount path as identity.

    HF origins already name an immutable repository/file. For local sources the
    original file is content-addressed; its historical URI remains provenance
    in the Parquet, but is not an identity component. This is row identity, not
    a claim that different HF/EOS productions contain distinct physics events.
    """
    result = pop_origin({PREFIX + key: value for key, value in origin.items()})
    if result["source"] == "local":
        result["file"] = "sha256:" + result["sha256"]
    return result


def event_id(origin: dict) -> str:
    payload = json.dumps(logical_origin(origin), sort_keys=True, separators=(",", ":"))
    return "origin-v1:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_columns(schema: pa.Schema) -> list[str]:
    """Reject incomplete/ambiguous provenance rather than replacing its origin."""
    present = {name for name in schema.names if name.startswith(PREFIX)}
    if not present:
        return []
    if present != set(ORIGIN_COLUMNS):
        raise ValueError("Incomplete or unknown reserved provenance columns.")
    for name, dtype in ORIGIN_FIELDS.items():
        if schema.field(PREFIX + name).type != dtype:
            raise ValueError(f"Invalid provenance type for {PREFIX + name}; expected {dtype}.")
    return ORIGIN_COLUMNS.copy()


def pop_origin(row: dict) -> dict | None:
    """Remove origin columns before feature conversion and validate their values."""
    if not any(name.startswith(PREFIX) for name in row):
        return None
    if any(name not in row for name in ORIGIN_COLUMNS):
        raise ValueError("Incomplete row provenance.")
    origin = {name: row.pop(PREFIX + name) for name in ORIGIN_FIELDS}
    for name in ("source", "file", "process", "sha256"):
        if not isinstance(origin[name], str) or not origin[name]:
            raise ValueError(f"Missing provenance {name}.")
    if origin["revision"] is not None and not isinstance(origin["revision"], str):
        raise ValueError("Invalid provenance revision.")
    if type(origin["row"]) is not int or origin["row"] < 0:
        raise ValueError("Origin row must be a nonnegative zero-based index.")
    if len(origin["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in origin["sha256"]):
        raise ValueError("Invalid origin SHA256.")
    return origin


def with_origin(table: pa.Table, origin: dict, row_indices: list[int]) -> pa.Table:
    """Attach origins once; preserve upstream origins on any subsequent export."""
    if provenance_columns(table.schema):
        for row in table.select(ORIGIN_COLUMNS).to_pylist():
            pop_origin(row)
        return table
    for name, dtype in ORIGIN_FIELDS.items():
        values = row_indices if name == "row" else [origin.get(name)] * len(table)
        table = table.append_column(PREFIX + name, pa.array(values, type=dtype))
    return table
