"""Schema inspection helpers for COLLIDE-2V parquet files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def inspect_schema(sample: Any) -> dict[str, str]:
    """Return a tiny type summary for smoke-level inspection."""

    if isinstance(sample, dict):
        return {key: type(value).__name__ for key, value in sample.items()}
    return {"sample_type": type(sample).__name__}


def inspect_parquet_schema(path: str | Path) -> dict[str, Any]:
    """Inspect one parquet file without reading its row data."""

    parquet_path = Path(path).expanduser()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    if not parquet_path.is_file():
        raise ValueError(f"Parquet path is not a file: {parquet_path}")

    schema = pq.read_schema(str(parquet_path))
    columns = list(schema.names)
    return {
        "path": str(parquet_path),
        "columns": columns,
        "num_columns": len(columns),
        "schema": str(schema),
    }


def inspect_dataset_schema(
    paths: list[str | Path],
    max_files: int | None = None,
) -> dict[str, Any]:
    """Inspect parquet schemas across several files.

    Unreadable files are kept in the report as error entries. The function only
    raises if no file can be read successfully.
    """

    if max_files is not None and max_files < 0:
        raise ValueError("max_files must be non-negative or None.")

    selected_paths = [Path(path).expanduser() for path in paths]
    if max_files is not None:
        selected_paths = selected_paths[:max_files]

    per_file: list[dict[str, Any]] = []
    readable_entries: list[dict[str, Any]] = []
    failed_entries: list[dict[str, Any]] = []

    for path in selected_paths:
        try:
            entry = inspect_parquet_schema(path)
            entry["error"] = None
            readable_entries.append(entry)
        except Exception as exc:  # noqa: BLE001 - report all file-level failures.
            entry = {
                "path": str(path),
                "columns": [],
                "num_columns": 0,
                "schema": None,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            failed_entries.append(entry)
        per_file.append(entry)

    if not readable_entries:
        errors = "; ".join(entry["error"] for entry in failed_entries) or "no files provided"
        raise RuntimeError(f"No readable parquet files found: {errors}")

    column_sets = [set(entry["columns"]) for entry in readable_entries]
    schema_strings = {entry["schema"] for entry in readable_entries}
    union_columns = sorted(set().union(*column_sets))

    warnings: list[str] = []
    if len(column_sets) > 1 and any(columns != column_sets[0] for columns in column_sets[1:]):
        warnings.append("Readable parquet files do not all have the same columns.")
    if len(schema_strings) > 1:
        warnings.append("Readable parquet files do not all have the same schema.")
    if failed_entries:
        warnings.append(f"{len(failed_entries)} parquet file(s) could not be read.")

    return {
        "num_files_requested": len(selected_paths),
        "num_files_inspected": len(readable_entries),
        "num_files_failed": len(failed_entries),
        "inspected_files": [str(path) for path in selected_paths],
        "readable_files": [entry["path"] for entry in readable_entries],
        "failed_files": [entry["path"] for entry in failed_entries],
        "per_file": per_file,
        "columns_by_file": {entry["path"]: entry["columns"] for entry in per_file},
        "schemas_by_file": {entry["path"]: entry["schema"] for entry in per_file},
        "union_columns": union_columns,
        "warnings": warnings,
    }


def write_schema_report(
    report: dict[str, Any],
    output_json: str | Path,
    output_txt: str | Path | None = None,
) -> None:
    """Write schema inspection reports as JSON and optional plain text."""

    json_path = Path(output_json).expanduser()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if output_txt is None:
        return

    txt_path = Path(output_txt).expanduser()
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(_format_text_report(report), encoding="utf-8")


def _format_text_report(report: dict[str, Any]) -> str:
    lines = [
        "FlowyForge COLLIDE-2V Schema Report",
        f"Files requested: {report.get('num_files_requested', 0)}",
        f"Files inspected: {report.get('num_files_inspected', 0)}",
        f"Files failed: {report.get('num_files_failed', 0)}",
        f"Union columns: {len(report.get('union_columns', []))}",
        "",
        "Columns:",
    ]
    union_columns = report.get("union_columns", [])
    lines.extend(f"- {column}" for column in union_columns)

    warnings = report.get("warnings", [])
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(["", "Files:"])
    for entry in report.get("per_file", []):
        lines.append(f"- {entry.get('path')}")
        if entry.get("error"):
            lines.append(f"  error: {entry['error']}")
            continue
        lines.append(f"  columns: {', '.join(entry.get('columns', []))}")
        lines.append("  schema:")
        schema = str(entry.get("schema", "")).splitlines()
        lines.extend(f"    {line}" for line in schema)

    return "\n".join(lines) + "\n"


__all__ = [
    "inspect_dataset_schema",
    "inspect_parquet_schema",
    "inspect_schema",
    "write_schema_report",
]
