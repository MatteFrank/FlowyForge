"""Bounded raw Hugging Face exports for the structured Perceiver pipeline."""

from __future__ import annotations

import json
import math
import re
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from flowyforge.data_plugins.collide_v2.provenance import sha256_file

MAX_ROWS_RAW_MODE = 100


def _limits(rows: int, files: int) -> None:
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS_RAW_MODE:
        raise ValueError(f"Raw HF materialization needs 1 <= max_rows <= {MAX_ROWS_RAW_MODE} per process.")
    if type(files) is not int or files < 1:
        raise ValueError("max_files must be a positive integer.")


def _empty_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing HF export: {path}. Choose an empty output directory.")
    path.mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def materialize_hf_collide1m_subset(
    dataset_name: str,
    split: str,
    output_dir: str | Path,
    max_rows: int = 20,
    max_files: int = 1,
    data_dir: str | None = None,
    data_files: str | list[str] | None = None,
    columns: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Preserve nested columns from the first bounded rows of a streamed process.

    This is a small raw export, not full-dataset downloading or uniform sampling
    over the remote repository. Infer one Arrow schema over the entire bounded
    sample so initially empty lists do not fix later chunks to null-only types.
    """
    _limits(max_rows, max_files)
    if not data_dir and data_files is None:
        raise ValueError("Select a process with data_dir or explicit data_files.")
    output = Path(output_dir).expanduser()
    _empty_output(output)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the Hugging Face dependencies with pip install -r requirements.txt") from exc
    kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if data_dir:
        kwargs["data_dir"] = data_dir
    if data_files is not None:
        kwargs["data_files"] = data_files
    if columns is not None:
        kwargs["columns"] = columns
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    rows = list(islice(load_dataset(dataset_name, **kwargs), max_rows))
    if not rows:
        raise ValueError(f"No events returned for HF process {data_dir or data_files}.")
    # HF rows can differ in optional fields; never silently discard a later key.
    names = list(dict.fromkeys(key for row in rows for key in row))
    table = pa.Table.from_pylist([{name: row.get(name) for name in names} for row in rows])
    rows_per_file = math.ceil(len(rows) / max_files)
    files = []
    for index, offset in enumerate(range(0, len(rows), rows_per_file)):
        name = f"part-{index:05d}.parquet"
        pq.write_table(table.slice(offset, rows_per_file), output / name)
        files.append(name)
    manifest = {
        "dataset_name": dataset_name, "split": split, "data_dir": data_dir,
        "data_files": data_files, "materialization_mode": "raw", "columns": names,
        "rows_written": len(rows), "parquet_files": files,
        "sampling": "first rows of remote stream, capped before export",
    }
    _write_manifest(output / "hf_subset_manifest.json", manifest)
    return manifest


def _safe_process_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "process"


def materialize_hf_collide1m_multi_process_subset(
    dataset_name: str,
    split: str,
    output_dir: str | Path,
    data_dirs: list[str],
    max_rows_per_process: int = 20,
    max_files_per_process: int = 1,
    columns: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export each process separately; partial failures are reported and raised."""
    _limits(max_rows_per_process, max_files_per_process)
    if not data_dirs or any(not isinstance(name, str) or not name.strip() for name in data_dirs):
        raise ValueError("Provide nonempty HF process directory names.")
    folders = [_safe_process_name(name) for name in data_dirs]
    if len({name.casefold() for name in folders}) != len(data_dirs):
        raise ValueError("HF process names must map to distinct output folders.")
    output = Path(output_dir).expanduser()
    _empty_output(output)
    counts, errors, files = {}, {}, []
    for name, folder in zip(data_dirs, folders):
        try:
            part = materialize_hf_collide1m_subset(
                dataset_name, split, output / folder, max_rows=max_rows_per_process,
                max_files=max_files_per_process, data_dir=name, columns=columns, cache_dir=cache_dir,
            )
            counts[name] = part["rows_written"]
            files.extend(f"{folder}/{filename}" for filename in part["parquet_files"])
        except Exception as exc:  # Preserve a usable failure manifest for remote IO errors.
            counts[name] = 0
            errors[name] = f"{type(exc).__name__}: {exc}"
    manifest = {
        "dataset_name": dataset_name, "split": split, "materialization_mode": "raw",
        "process_folders": dict(zip(data_dirs, folders)), "rows_written_per_process": counts,
        "parquet_files": files, "errors": errors,
        "processes_materialized": sum(count > 0 for count in counts.values()),
    }
    _write_manifest(output / "hf_multi_subset_manifest.json", manifest)
    if errors:
        raise RuntimeError(f"Incomplete HF export; inspect {output / 'hf_multi_subset_manifest.json'}: {errors}")
    return manifest


def prepare_hf_source(config: Mapping[str, Any]) -> dict[str, Any]:
    """Prepare the raw source described by configs/paths/hf_collide1m_raw.yaml."""
    if unknown := set(config) - {"paths", "data"}:
        raise ValueError(f"Unknown preparation sections: {sorted(unknown)}. Use the raw-only HF config.")
    paths, data = config["paths"], config.get("data", {})
    if paths.get("dataset_backend") != "hf":
        raise ValueError("Preparation is only needed for the hf backend; mounted Parquet goes directly to run_classification.py.")
    allowed_paths = {"dataset_backend", "hf_dataset_name", "hf_split", "hf_data_dirs", "dataset_dir", "local_cache_dir"}
    if unknown := set(paths) - allowed_paths:
        raise ValueError(f"Unknown preparation paths: {sorted(unknown)}")
    if unknown := set(data) - {"max_rows_per_process", "max_files"}:
        raise ValueError(f"Unknown preparation data settings: {sorted(unknown)}. Only raw nested exports are supported.")
    directories = paths.get("hf_data_dirs")
    if not isinstance(directories, list) or len(directories) < 2:
        raise ValueError("Configure at least two hf_data_dirs for classification.")
    rows = data.get("max_rows_per_process", 20)
    _limits(rows, data.get("max_files", 1))
    if rows < 3:
        raise ValueError("Classification requires at least 3 events per process.")
    manifest = materialize_hf_collide1m_multi_process_subset(
        paths["hf_dataset_name"], paths.get("hf_split", "train"), paths["dataset_dir"], directories,
        max_rows_per_process=rows, max_files_per_process=data.get("max_files", 1),
        cache_dir=paths.get("local_cache_dir"),
    )
    if any(count < 3 for count in manifest["rows_written_per_process"].values()):
        raise ValueError("HF export contains fewer than 3 events in a process; inspect the export manifest.")
    return manifest


def plan_hf_files(dataset: str, revision: str, processes: Mapping[str, list[str]],
                  max_download_bytes: int, *, api=None) -> dict[str, Any]:
    """Verify explicit files at a pinned commit; inspect metadata, download no data.

    The cap is the total compressed size of the selected files, including cached
    ones. It is not a bound on retries, HTTP overhead or exported/decompressed data.
    """
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("HF revision must be a pinned full 40-character commit SHA, not main/tag.")
    if type(max_download_bytes) is not int or max_download_bytes < 1:
        raise ValueError("max_download_bytes must be a positive integer.")
    if not processes or any(not isinstance(name, str) or not name for name in processes):
        raise ValueError("Provide named HF processes and explicit files.")
    selected = []
    for process, filenames in processes.items():
        if not isinstance(filenames, list) or not filenames:
            raise ValueError(f"{process}: provide an explicit nonempty list of HF files.")
        for name in filenames:
            if (not isinstance(name, str) or PurePosixPath(name).is_absolute()
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                    or any(char in name for char in "*?[]\\:") or not name.endswith(".parquet")
                    or name.split("/", 1)[0] != process):
                raise ValueError(f"Invalid explicit HF file for process {process}: {name!r}")
            selected.append((process, name))
    if len({name for _, name in selected}) != len(selected):
        raise ValueError("Duplicate HF files are not allowed.")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.dataset_info(dataset, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("HF resolved revision differs from the pinned commit.")
    available = {entry.rfilename: entry for entry in info.siblings}
    files = []
    for process, name in selected:
        if name not in available:
            raise ValueError(f"HF file not found at {revision}: {name}")
        entry = available[name]
        if type(entry.size) is not int or entry.size <= 0:
            raise ValueError(f"Missing trustworthy remote size for {name}; refusing download.")
        lfs = entry.lfs
        digest = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files.append({"process": process, "logical_file": name, "size_bytes": entry.size,
                      "remote_sha256": digest})
    total = sum(entry["size_bytes"] for entry in files)
    if total > max_download_bytes:
        raise ValueError(f"Selected HF files require {total} bytes, exceeding max_download_bytes={max_download_bytes}. No download started.")
    return {"kind": "hf_files", "dataset": dataset, "revision": revision,
            "files": files, "selected_bytes": total, "max_download_bytes": max_download_bytes,
            "sampling_scope": "explicitly selected files, not a uniform sample of the remote dataset"}


def download_hf_files(plan: Mapping[str, Any], download_dir: str | Path, *, downloader=None) -> list[dict]:
    """Retain only planned files locally; verify sizes/hashes and protect originals."""
    if sum(entry["size_bytes"] for entry in plan["files"]) > plan["max_download_bytes"]:
        raise ValueError("HF download plan exceeds its byte limit.")
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    root = Path(download_dir).expanduser().resolve() / plan["revision"]
    results = []
    for entry in plan["files"]:
        path = root / entry["logical_file"]
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"HF destination escapes download directory: {path}")
        expected_hash = entry.get("remote_sha256")
        if path.exists():
            if (not path.is_file() or path.stat().st_size != entry["size_bytes"]
                    or not expected_hash or sha256_file(path) != expected_hash):
                raise FileExistsError(f"Refusing to overwrite unverifiable/different cached original: {path}")
            print(f"Reusing verified original: {entry['logical_file']}", flush=True)
        else:
            print(f"Downloading {entry['logical_file']} ({entry['size_bytes']:,} bytes)", flush=True)
            downloaded = Path(downloader(repo_id=plan["dataset"], filename=entry["logical_file"],
                                        repo_type="dataset", revision=plan["revision"], local_dir=str(root)))
            if downloaded.resolve() != path.resolve():
                raise ValueError("Downloader returned an unexpected path.")
        digest = sha256_file(path)
        if path.stat().st_size != entry["size_bytes"] or (expected_hash and digest != expected_hash):
            raise ValueError(f"Downloaded file failed size/SHA256 verification: {path}")
        results.append({**entry, "path": str(path), "sha256": digest})
    return results


__all__ = ["materialize_hf_collide1m_subset", "materialize_hf_collide1m_multi_process_subset",
           "prepare_hf_source", "plan_hf_files", "download_hf_files"]
