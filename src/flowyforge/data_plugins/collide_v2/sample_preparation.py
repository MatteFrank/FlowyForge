"""Prepare reproducible two-view raw samples from selected local or HF files."""

from __future__ import annotations

import base64
import bisect
from contextlib import closing
import json
import os
import shutil
import time
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from flowyforge.core.config import load_yaml
from flowyforge.core.parallel import map_files, peak_rss_bytes
from flowyforge.data_plugins.collide_v2.hf_collide1m import (
    _safe_process_name, download_hf_files, plan_hf_files,
)
from flowyforge.data_plugins.collide_v2.provenance import ORIGIN_COLUMNS, provenance_columns, sha256_file, with_origin
from flowyforge.data_plugins.collide_v2.streaming import (
    _check_file, _reader_config, _resolve_processes, _schema_info, _schema_digest, _event_records,
    build_manifest, inspect_dataset, create_partition_manifest, save_partition_manifest, _policy, _audit_warnings, _split_counts,
)

VIEWS = ("FullReco", "L1T")
MANIFEST_NAME = "sample_manifest.json"


def _partition_settings(value=None):
    settings = {"enabled": True, "check_integrity_SHA256": "fast", "validation_fraction": .2,
                "test_fraction": .2, "read_batch_size": 2048, "num_workers": 4, **(value or {})}
    _known(settings, {"enabled", "check_integrity_SHA256", "validation_fraction", "test_fraction", "read_batch_size", "num_workers"}, "partitions")
    _policy(settings["check_integrity_SHA256"])
    if type(settings["enabled"]) is not bool:
        raise ValueError("partitions.enabled must be boolean.")
    if type(settings["num_workers"]) is not int or settings["num_workers"] not in (1, 2, 4, 8):
        raise ValueError("partitions.num_workers must be one of 1, 2, 4, 8.")
    if type(settings["read_batch_size"]) is not int or settings["read_batch_size"] < 1:
        raise ValueError("partitions.read_batch_size must be a positive integer.")
    vf, tf = settings["validation_fraction"], settings["test_fraction"]
    if not (0 < vf < 1 and 0 < tf < 1 and vf + tf < 1):
        raise ValueError("Partition fractions must be positive and sum to less than one.")
    return settings


def _known(value: Mapping, allowed: set[str], section: str) -> None:
    if unknown := set(value) - allowed:
        raise ValueError(f"Unknown {section} options: {sorted(unknown)}")


def load_sample_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    raw = load_yaml(path)
    _known(raw, {"source", "output_dir", "selection", "model_config", "reader", "partitions"}, "sample")
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
            "model_config": str(model_path.resolve()), "readers": readers,
            "partitions": _partition_settings(raw.get("partitions"))}


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
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _metadata_bytes(metadata: dict | None) -> dict[str, str]:
    return {base64.b64encode(key).decode("ascii"): base64.b64encode(value).decode("ascii")
            for key, value in (metadata or {}).items()}


def _export_file(info: dict, selected: list[int] | None, destination: Path, origin: dict,
                 read_batch_size: int, *, policy="fast", collect=False, timing=None) -> tuple[int, list]:
    """Copy Arrow values/schema unchanged; select rows before adding provenance."""
    start, stop = info["offset"], info["offset"] + info["num_rows"]
    chosen = None if selected is None else selected[bisect.bisect_left(selected, start):bisect.bisect_left(selected, stop)]
    if chosen == [] or info["num_rows"] == 0:
        return 0, []
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer, exported, row_start, records = None, 0, 0, []
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
                        if collect:
                            collection_started = time.perf_counter()
                            records.extend(_event_records(table, origin["process"], policy=policy, timing=timing))
                            if timing is not None:
                                timing["partition_collection_seconds"] += time.perf_counter() - collection_started
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
    return exported, records


def _validate_file(job):
    """Full existing reader audit for one file, in one worker and bounded batches."""
    path, name, readers, read_batch_size, checksum = job
    path = Path(path)
    stat = path.stat()
    reports = {}
    with pq.ParquetFile(path) as parquet:
        groups = [parquet.metadata.row_group(i).num_rows for i in range(parquet.num_row_groups)]
        for view, reader in readers.items():
            info = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "num_rows": parquet.metadata.num_rows, "offset": 0, "task_offset": 0,
                    "row_groups": groups, **_schema_info(parquet.schema_arrow, reader, path)}
            if checksum is not None:
                info["sha256"] = checksum
            process = {"name": name, "label": 0, "files": [info], "num_rows": info["num_rows"],
                       "selected_rows": info["num_rows"], "selected_indices": None}
            reports[view] = inspect_dataset({"seed": 0, "reader": reader, "processes": [process], "split_counts": {}},
                audit_events_per_class=max(1, info["num_rows"]), read_batch_size=read_batch_size)
    return reports


def _merge_reports(parts, manifest):
    """Combine the same exhaustive statistics without retaining event tensors."""
    result = {}
    for view in VIEWS:
        reports = [part[view] for part in parts]
        report = {**reports[0], "processes": {}, "collections": {}, "fields": {}, "warnings": []}
        for key in ("num_events", "num_files", "unit_metadata_fields", "unit_assumed_fields"):
            report[key] = sum(part[key] for part in reports)
        for part in reports:
            report["warnings"].extend(part["warnings"])
            for name, stats in part["processes"].items():
                totals = report["processes"].setdefault(name, dict.fromkeys(stats, 0))
                for key, value in stats.items():
                    totals[key] += value
            for name, stats in part["collections"].items():
                totals = report["collections"].setdefault(name, {})
                for key, value in stats.items():
                    totals[key] = sorted(set(totals.get(key, [])) | set(value)) if isinstance(value, list) else totals.get(key, 0) + value
            for name, stats in part["fields"].items():
                totals = report["fields"].setdefault(name, {})
                for key, value in stats.items():
                    if key in ("min", "max"):
                        values = [v for v in (totals.get(key), value) if v is not None]
                        totals[key] = (min(values) if key == "min" else max(values)) if values else None
                    else:
                        totals[key] = totals.get(key, 0) + value
        report["warnings"] = _audit_warnings(report)
        settings = manifest.get("partitions", {})
        counts = [_split_counts(p["exported_rows"], settings.get("validation_fraction", .2),
                                settings.get("test_fraction", .2)) for p in manifest["processes"].values()]
        report["split_counts"] = {split: sum(c[split] for c in counts) for split in ("train", "validation", "test")}
        report["scope"] = "all exported rows and file schemas; existing reader transformations; original source rows outside the sample are not audited"
        result[view] = report
    return result


def _prepared_files(output, manifest, policy):
    """Validate completed inventory/footer metadata; fast never hashes payloads."""
    processes = {name: [] for name in manifest["processes"]}
    counts = dict.fromkeys(processes, 0)
    seen = set()
    for entry in manifest["output_files"]:
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or path in seen:
            raise ValueError("Output file escapes sample directory or is duplicated.")
        seen.add(path)
        stat = path.stat()
        if stat.st_size != entry["size_bytes"]:
            raise ValueError(f"Prepared file changed: {path}")
        if policy == "strict" and entry.get("sha256") and sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Prepared file changed: {path}")
        if policy == "fast" and "mtime_ns" in entry and stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Prepared file changed (mtime): {path}")
        with pq.ParquetFile(path) as parquet:
            if parquet.metadata.num_rows != entry["num_rows"]:
                raise ValueError(f"Prepared row count changed: {path}")
            if not provenance_columns(parquet.schema_arrow):
                raise ValueError(f"Prepared file lacks persistent provenance: {path}")
            if "schema_sha256" in entry and _schema_digest(parquet.schema_arrow) != entry["schema_sha256"]:
                raise ValueError(f"Prepared schema changed: {path}")
            for reader in manifest["readers"].values():
                _schema_info(parquet.schema_arrow, reader, path)
        processes[entry["process"]].append(str(path))
        counts[entry["process"]] += entry["num_rows"]
    if any(count != manifest["processes"][name]["exported_rows"] for name, count in counts.items()):
        raise ValueError("Prepared inventory row counts differ from sample manifest.")
    resolved = _resolve_processes({"processes": {name: output / name for name in processes}})
    if {p for paths in resolved.values() for p in paths} != seen:
        raise ValueError("Prepared file inventory changed (missing/extra files).")
    return {"processes": processes}


def inspect_prepared_sample(output_dir: str | Path) -> dict:
    """Read-only, exhaustive two-view audit under the sample's recorded policy."""
    output = Path(output_dir).expanduser().resolve()
    manifest = _completed_sample(output)
    policy = manifest.get("check_integrity_SHA256", "strict")
    _prepared_files(output, manifest, policy)
    jobs = [(str(output / f["path"]), f["process"], manifest["readers"],
             manifest["selection"]["read_batch_size"], f.get("sha256") if policy == "strict" else None)
            for f in manifest["output_files"]]
    return _merge_reports(list(map_files(_validate_file, jobs, manifest.get("partitions", {}).get("num_workers", 1))), manifest)


def _completed_sample(output):
    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Sample manifest must be a complete version-1 export.")
    return manifest


def prepare_partitions(config):
    """Add immutable partitions to an existing complete sample, without raw export.

    Fast reads only provenance columns/footer metadata. Existing strict sample
    checksums remain untouched; choosing fast is an explicit weaker verification.
    """
    output = Path(config["output_dir"])
    target = output / "partitions.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing partitions: {target}")
    started = time.perf_counter()
    settings = _partition_settings(config.get("partitions"))
    manifest = _completed_sample(output)
    policy = settings["check_integrity_SHA256"]
    source = _prepared_files(output, manifest, policy)
    if set(source["processes"]) != set(config["source"]["processes"]):
        raise ValueError("Configured processes differ from completed prepared sample.")
    verification = time.perf_counter() - started
    timing = {"event_fingerprinting_seconds": 0., "file_sha256_seconds": 0.}
    partitions = create_partition_manifest(source, seed=config["selection"]["seed"],
        **{k: v for k, v in settings.items() if k != "enabled"}, performance=timing)
    save_partition_manifest(partitions, target)
    _write_json(output / "partition_performance.json", {**timing, "check_integrity_SHA256": policy,
        "num_workers": settings["num_workers"], "prepared_verification_seconds": verification,
        "total_wall_seconds": time.perf_counter() - started, "parent_peak_rss_bytes": peak_rss_bytes(),
        "scope": "completed prepared sample only; raw export and physical reader audit not repeated"})
    return partitions


def _prepare_file(job):
    info, selected, description, destination, source, revision, readers, read_batch_size, settings = job
    timing = {"source_verification_seconds": 0., "export_seconds": 0., "partition_collection_seconds": 0.,
              "event_fingerprinting_seconds": 0., "parquet_validation_seconds": 0., "output_sha256_seconds": 0.}
    policy = settings["check_integrity_SHA256"]
    destination = Path(destination)
    temporary = destination.with_suffix(".parquet.tmp")
    _check_file(info)
    started = time.perf_counter()
    with pq.ParquetFile(info["path"]) as parquet:
        has_origins = bool(provenance_columns(parquet.schema_arrow))
        digest = description.get("sha256")
        if digest is None and (policy == "strict" or not has_origins):
            digest = sha256_file(info["path"])
        description = {**description, "num_rows": info["num_rows"], "mtime_ns": info["mtime_ns"],
            "row_groups": info["row_groups"], "schema_metadata_base64": _metadata_bytes(parquet.schema_arrow.metadata),
            "parquet_metadata_base64": _metadata_bytes(parquet.metadata.metadata)}
        if digest is not None:
            description["sha256"] = digest
    timing["source_verification_seconds"] = time.perf_counter() - started
    origin = {"source": "hf://datasets/" + source["dataset"] if source["kind"] == "hf_files" else "local",
              "revision": revision, "file": description["logical_file"], "process": description["process"], "sha256": digest}
    started = time.perf_counter()
    # Count only the metadata collection separately; export_seconds excludes it.
    collect_timing = {"event_fingerprinting_seconds": 0., "partition_collection_seconds": 0.}
    try:
        count, records = _export_file(info, selected, temporary, origin, read_batch_size,
            policy=policy, collect=settings["enabled"], timing=collect_timing)
        timing["export_seconds"] = time.perf_counter() - started - collect_timing["partition_collection_seconds"]
        timing.update(collect_timing)
        description["exported_rows"] = count
        if not count:
            return description, None, [], None, {**timing, "pid": os.getpid(), "peak_rss_bytes": peak_rss_bytes()}
        temporary.replace(destination)
        stat = destination.stat()
        with pq.ParquetFile(destination) as parquet:
            signature = _schema_digest(parquet.schema_arrow)
        entry = {"process": description["process"], "path": destination.name,
                 "input_file": description["logical_file"], "num_rows": count,
                 "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "schema_sha256": signature}
        if policy == "strict":
            started = time.perf_counter()
            entry["sha256"] = sha256_file(destination)
            timing["output_sha256_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        reports = _validate_file((str(destination), description["process"], readers, read_batch_size, entry.get("sha256")))
        timing["parquet_validation_seconds"] = time.perf_counter() - started
        _check_file(info)
        if destination.stat().st_mtime_ns != stat.st_mtime_ns:
            raise ValueError(f"Prepared file changed during validation: {destination}")
        return description, entry, records, reports, {**timing, "pid": os.getpid(), "peak_rss_bytes": peak_rss_bytes()}
    finally:
        temporary.unlink(missing_ok=True)


def prepare_sample(config: Mapping[str, Any]) -> dict:
    """Export, audit both views, and publish partitions in one protected workflow."""
    started = time.perf_counter()
    output = Path(config["output_dir"])
    settings = _partition_settings(config.get("partitions"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty sample directory: {output}")
    plan = plan_sample(config)
    discovery = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "status": "preparing", "source": plan,
        "check_integrity_SHA256": settings["check_integrity_SHA256"], "partitions": settings,
        "selection": {**config["selection"], "method": "uniform without replacement within selected files per process; sorted source rows on export",
                      "scope": "selected files only; not uniform over the complete COLLIDE repository"},
        "readers": config["readers"], "processes": {}, "output_files": [],
        "provenance": {"columns": ORIGIN_COLUMNS, "row_index_base": 0,
            "policy": "preserve existing origins on re-export; local originals require one initial file SHA256 for origin-v1"}}
    with (output / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    _write_json(output / "preparation_config.json", config)
    performance = {"discovery_seconds": discovery, "num_workers": settings["num_workers"], "workers": {}, "file_timings": []}
    try:
        source = config["source"]
        phase = time.perf_counter()
        inputs = (download_hf_files(plan, source["download_dir"]) if source["kind"] == "hf_files" else plan["files"])
        performance["download_verification_seconds"] = time.perf_counter() - phase
        paths = {name: [] for name in source["processes"]}
        descriptions = {}
        for entry in inputs:
            path = Path(entry["path"]).resolve()
            paths[entry["process"]].append(str(path))
            descriptions[str(path)] = {**entry, "path": str(path)}
        inventory_started = time.perf_counter()
        inventory = build_manifest({"processes": paths, "max_events_per_class": config["selection"]["events_per_process"]},
            seed=config["selection"]["seed"], **config["readers"]["FullReco"])
        performance["discovery_seconds"] += time.perf_counter() - inventory_started
        collected, jobs = {}, []
        for process in inventory["processes"]:
            name = process["name"]
            if process["selected_rows"] < config["selection"]["events_per_process"]:
                raise ValueError(f"{name}: only {process['num_rows']} rows available; add files or reduce events_per_process.")
            manifest["processes"][name] = {"available_rows_in_selected_files": process["num_rows"], "exported_rows": 0}
            collected[name] = {"name": name, "files": [], "events": []}
            for index, info in enumerate(process["files"]):
                jobs.append((info, process["selected_indices"], descriptions[info["path"]],
                    str(output / name / f"part-{index:05d}.parquet"), source, plan["revision"], config["readers"],
                    config["selection"]["read_batch_size"], settings))
        manifest["source"]["files"], reports = [], []
        phase = time.perf_counter()
        with closing(map_files(_prepare_file, jobs, settings["num_workers"])) as results:
            for description, entry, records, report, timing in results:
                name = description["process"]
                manifest["source"]["files"].append(description)
                performance["file_timings"].append({"file": description["logical_file"], **timing})
                performance["workers"][str(timing["pid"])] = timing["peak_rss_bytes"]
                if entry is None:
                    continue
                relative = (Path(name) / entry["path"]).as_posix()
                manifest["output_files"].append({**entry, "path": relative})
                manifest["processes"][name]["exported_rows"] += entry["num_rows"]
                reports.append(report)
                if settings["enabled"]:
                    info = {"size_bytes": entry["size_bytes"], "num_rows": entry["num_rows"], "event_ids": [e["id"] for e in records]}
                    if settings["check_integrity_SHA256"] == "strict":
                        info["sha256"] = entry["sha256"]
                    else:
                        info.update(name=entry["path"], mtime_ns=entry["mtime_ns"], schema_sha256=entry["schema_sha256"])
                    collected[name]["files"].append(info)
                    collected[name]["events"].extend(records)
        performance["file_processing_wall_seconds"] = time.perf_counter() - phase
        if any(p["exported_rows"] != config["selection"]["events_per_process"] for p in manifest["processes"].values()):
            raise ValueError("Export row count mismatch.")
        reports = _merge_reports(reports, manifest)
        _write_json(output / "validation.json", reports)
        phase = time.perf_counter()
        if settings["enabled"]:
            partitions = create_partition_manifest({}, seed=config["selection"]["seed"],
                **{k: v for k, v in settings.items() if k != "enabled"}, _collected=list(collected.values()))
            save_partition_manifest(partitions, output / "partitions.json")
            manifest["partition_manifest"] = {"path": "partitions.json", "sha256": partitions["sha256"], "version": partitions["version"]}
        performance["partition_generation_seconds"] = time.perf_counter() - phase
        manifest["status"] = "complete"
        manifest["validation"] = {view: {"events_checked": report["num_events"], "warnings": report["warnings"]} for view, report in reports.items()}
    except BaseException as exc:
        # Waited-for workers have stopped before cleanup. Never leave a partially
        # exported dataset that looks complete; retain failure/config diagnostics.
        for name in manifest["processes"]:
            shutil.rmtree(output / name, ignore_errors=True)
        (output / "partitions.json").unlink(missing_ok=True)
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", output_files=[])
        _write_json(output / MANIFEST_NAME, manifest)
        raise
    performance["total_wall_seconds"] = time.perf_counter() - started
    performance["parent_peak_rss_bytes"] = peak_rss_bytes()
    performance["memory_scope"] = "per-process lifetime RSS peaks; not simultaneous aggregate job memory"
    performance["events_per_second"] = sum(p["exported_rows"] for p in manifest["processes"].values()) / performance["total_wall_seconds"]
    manifest["performance"] = performance
    _write_json(output / "performance.json", performance)
    _write_json(output / MANIFEST_NAME, manifest)
    return manifest
