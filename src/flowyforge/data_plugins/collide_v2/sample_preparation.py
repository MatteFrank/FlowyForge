"""Prepare reproducible two-view raw samples from selected local or HF files."""

from __future__ import annotations

import base64
import bisect
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from flowyforge.core.config import load_yaml
from flowyforge.data_plugins.collide_v2.hf_collide1m import (
    _safe_process_name, download_hf_files, plan_hf_files,
)
from flowyforge.data_plugins.collide_v2.provenance import ORIGIN_COLUMNS, sha256_file, with_origin
from flowyforge.data_plugins.collide_v2.streaming import (
    _check_file, _reader_config, _resolve_processes, _schema_info,
    build_manifest, inspect_dataset,
)

VIEWS = ("FullReco", "L1T")
MANIFEST_NAME = "sample_manifest.json"


def _known(value: Mapping, allowed: set[str], section: str) -> None:
    if unknown := set(value) - allowed:
        raise ValueError(f"Unknown {section} options: {sorted(unknown)}")


def load_sample_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    raw = load_yaml(path)
    _known(raw, {"source", "output_dir", "selection", "model_config", "reader"}, "sample")
    source = dict(raw["source"])
    kind = source.get("kind")
    if kind not in {"local", "hf_files"}:
        raise ValueError("source.kind must be local or hf_files.")
    allowed = {"kind", "processes"}
    if kind == "hf_files":
        allowed |= {"dataset", "revision", "download_dir", "max_download_bytes"}
    _known(source, allowed, "source")
    names = source.get("processes", {})
    if (not isinstance(names, dict) or len(names) < 2
            or any(not isinstance(name, str) or _safe_process_name(name) != name for name in names)
            or len({name.casefold() for name in names}) != len(names)):
        raise ValueError("Provide at least two distinct, filesystem-safe process names.")
    selection = {"events_per_process": 5000, "seed": 42, "read_batch_size": 64,
                 **raw.get("selection", {})}
    _known(selection, {"events_per_process", "seed", "read_batch_size"}, "selection")
    for key, minimum in (("events_per_process", 3), ("seed", 0), ("read_batch_size", 1)):
        if type(selection[key]) is not int or selection[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    model_path = Path(raw["model_config"]).expanduser()
    if not model_path.is_absolute():
        model_path = path.parent / model_path
    model_data = load_yaml(model_path).get("data", {})
    reader = {"units": {"momentum": "GeV", "angle": "rad"}, **model_data, **raw.get("reader", {})}
    _known(reader, {"units", "collection_limits", "eta_clip", "value_clip"}, "reader")
    # Resolve exactly the same object contract and units as the streaming reader.
    readers = {view: _reader_config(view=view, **reader) for view in VIEWS}
    output = Path(raw["output_dir"]).expanduser().resolve()
    if kind == "hf_files":
        download = Path(source["download_dir"]).expanduser().resolve()
        if output.is_relative_to(download) or download.is_relative_to(output):
            raise ValueError("download_dir and output_dir must be separate directories, not nested.")
        source["download_dir"] = str(download)
    return {"source": source, "output_dir": str(output), "selection": selection,
            "model_config": str(model_path.resolve()), "readers": readers}


def plan_sample(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory only: no downloads, exports, training or writes."""
    source = config["source"]
    if source["kind"] == "hf_files":
        return plan_hf_files(source["dataset"], source["revision"], source["processes"],
                             source["max_download_bytes"])
    files = []
    for process, paths in _resolve_processes(source).items():
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                for reader in config["readers"].values():
                    _schema_info(parquet.schema_arrow, reader, path)
                files.append({"process": process, "logical_file": path.as_uri(), "path": str(path),
                              "size_bytes": path.stat().st_size, "num_rows": parquet.metadata.num_rows})
    return {"kind": "local", "revision": None, "files": files,
            "selected_bytes": sum(entry["size_bytes"] for entry in files),
            "sampling_scope": "resolved local files only, not a uniform sample of an external dataset"}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _metadata_bytes(metadata: dict | None) -> dict[str, str]:
    return {base64.b64encode(key).decode("ascii"): base64.b64encode(value).decode("ascii")
            for key, value in (metadata or {}).items()}


def _export_file(info: dict, selected: list[int] | None, destination: Path, origin: dict,
                 read_batch_size: int) -> int:
    """Copy Arrow values/schema unchanged; select rows before adding provenance."""
    start, stop = info["offset"], info["offset"] + info["num_rows"]
    chosen = None if selected is None else selected[bisect.bisect_left(selected, start):bisect.bisect_left(selected, stop)]
    if chosen == [] or info["num_rows"] == 0:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer, exported, row_start = None, 0, 0
    _check_file(info)
    try:
        with pq.ParquetFile(info["path"]) as parquet:
            for group in range(parquet.num_row_groups):
                group_size = parquet.metadata.row_group(group).num_rows
                if chosen is not None and bisect.bisect_left(chosen, start + row_start) == bisect.bisect_left(chosen, start + row_start + group_size):
                    row_start += group_size
                    continue
                for batch in parquet.iter_batches(batch_size=read_batch_size, row_groups=[group], use_threads=False):
                    first, last = start + row_start, start + row_start + len(batch)
                    indices = list(range(row_start, row_start + len(batch))) if chosen is None else [
                        index - start for index in chosen[bisect.bisect_left(chosen, first):bisect.bisect_left(chosen, last)]]
                    if indices:
                        table = pa.Table.from_batches([batch], schema=parquet.schema_arrow)
                        table = table.take(pa.array([index - row_start for index in indices], type=pa.int64()))
                        table = with_origin(table, origin, indices)
                        if writer is None:
                            # Preserve original nested field names, types and metadata.
                            writer = pq.ParquetWriter(destination, table.schema, compression="zstd",
                                                      use_compliant_nested_type=False)
                            # Footer-only custom metadata may not be represented
                            # in schema_arrow. Regenerate ARROW:schema so the new
                            # origin columns remain visible to Arrow consumers.
                            writer.add_key_value_metadata({
                                key: value for key, value in (parquet.metadata.metadata or {}).items()
                                if key != b"ARROW:schema"
                            })
                        writer.write_table(table)
                        exported += len(table)
                    row_start += len(batch)
        _check_file(info)
    finally:
        if writer is not None:
            writer.close()
    return exported


def _validate_export(output: Path, manifest: dict) -> dict:
    processes = {name: [] for name in manifest["processes"]}
    for entry in manifest["output_files"]:
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Output file escapes sample directory.")
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Prepared file changed: {path}")
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != entry["num_rows"]:
                raise ValueError(f"Prepared row count changed: {path}")
            if not set(ORIGIN_COLUMNS).issubset(parquet.schema_arrow.names):
                raise ValueError(f"Prepared file lacks persistent provenance: {path}")
        processes[entry["process"]].append(str(path))
    reports = {}
    for view in VIEWS:
        print(f"Validating all exported rows through the {view} reader...", flush=True)
        inventory = build_manifest({"processes": processes}, seed=manifest["selection"]["seed"],
                                   **manifest["readers"][view])
        for process in inventory["processes"]:
            if process["num_rows"] != manifest["processes"][process["name"]]["exported_rows"]:
                raise ValueError(f"Unexpected exported count for {process['name']}")
        reports[view] = inspect_dataset(
            inventory, audit_events_per_class=max(p["num_rows"] for p in inventory["processes"]),
            read_batch_size=manifest["selection"]["read_batch_size"],
        )
        reports[view]["scope"] = "all exported rows and file schemas; existing reader transformations; original source rows outside the sample are not audited"
    return reports


def inspect_prepared_sample(output_dir: str | Path) -> dict:
    """Read-only checksum/schema/value/provenance validation of both views."""
    output = Path(output_dir).expanduser().resolve()
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Sample manifest must be a complete version-1 export.")
    return _validate_export(output, manifest)


def prepare_sample(config: Mapping[str, Any]) -> dict:
    """Prepare one protected raw sample, validating both views before success."""
    output = Path(config["output_dir"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty sample directory: {output}")
    plan = plan_sample(config)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "status": "preparing", "source": plan,
                "selection": {**config["selection"], "method": "uniform without replacement within selected files per process; sorted source rows on export",
                              "scope": "selected files only; not uniform over the complete COLLIDE repository"},
                "readers": config["readers"], "processes": {}, "output_files": [],
                "provenance": {"columns": ORIGIN_COLUMNS, "row_index_base": 0,
                               "policy": "preserve existing origins on re-export; never infer origins of historical exports"}}
    # Exclusive creation also protects against concurrent preparation jobs.
    with (output / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    _write_json(output / "preparation_config.json", config)
    try:
        source = config["source"]
        inputs = (download_hf_files(plan, source["download_dir"]) if source["kind"] == "hf_files"
                  else plan["files"])
        paths = {name: [] for name in source["processes"]}
        descriptions = {}
        for entry in inputs:
            path = Path(entry["path"]).resolve()
            entry = {**entry, "path": str(path)}
            paths[entry["process"]].append(str(path))
            descriptions[str(path)] = entry
        inventories = {view: build_manifest(
            {"processes": paths, "max_events_per_class": config["selection"]["events_per_process"]},
            seed=config["selection"]["seed"], **config["readers"][view],
        ) for view in VIEWS}
        inventory = inventories["FullReco"]
        for process in inventory["processes"]:
            if process["selected_rows"] < config["selection"]["events_per_process"]:
                raise ValueError(f"{process['name']}: only {process['num_rows']} rows available; add explicit files or reduce events_per_process.")
        manifest["source"]["files"] = []
        for process in inventory["processes"]:
            name, exported = process["name"], 0
            manifest["processes"][name] = {"available_rows_in_selected_files": process["num_rows"],
                                           "exported_rows": 0}
            for index, info in enumerate(process["files"]):
                description = descriptions[info["path"]]
                _check_file(info)
                digest = description.get("sha256") or sha256_file(info["path"])
                with pq.ParquetFile(info["path"]) as parquet:
                    description.update(sha256=digest, num_rows=info["num_rows"], mtime_ns=info["mtime_ns"],
                                       row_groups=info["row_groups"],
                                       schema_metadata_base64=_metadata_bytes(parquet.schema_arrow.metadata),
                                       parquet_metadata_base64=_metadata_bytes(parquet.metadata.metadata))
                manifest["source"]["files"].append(description)
                origin = {"source": "hf://datasets/" + source["dataset"] if source["kind"] == "hf_files" else "local",
                          "revision": plan["revision"], "file": description["logical_file"],
                          "process": name, "sha256": digest}
                relative = Path(name) / f"part-{index:05d}.parquet"
                count = _export_file(info, process["selected_indices"], output / relative, origin,
                                     config["selection"]["read_batch_size"])
                description["exported_rows"] = count
                if count:
                    manifest["output_files"].append({"process": name, "path": relative.as_posix(),
                        "input_file": description["logical_file"], "num_rows": count,
                        "size_bytes": (output / relative).stat().st_size, "sha256": sha256_file(output / relative)})
                exported += count
            manifest["processes"][name]["exported_rows"] = exported
            if exported != process["selected_rows"]:
                raise ValueError(f"Export row count mismatch for {name}.")
        _write_json(output / MANIFEST_NAME, manifest)
        reports = _validate_export(output, manifest)
        _write_json(output / "validation.json", reports)
        manifest["status"] = "complete"
        manifest["validation"] = {view: {"events_checked": report["num_events"], "warnings": report["warnings"]}
                                  for view, report in reports.items()}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(output / MANIFEST_NAME, manifest)
        raise
    _write_json(output / MANIFEST_NAME, manifest)
    return manifest
