#!/usr/bin/env python
"""Benchmark preparation + partitions on identical local inputs, without training.

Each case runs in a fresh process. Outputs are retained below a NEW directory;
no datasets/checkpoints are deleted. Optional git archive baseline runs the actual
older code (no checkout/reset). Timings include pool startup, exclude Python import
startup, and are warm/cache-order dependent. Worker phase sums are not wall times.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time


def _child(config_path, legacy):
    import resource
    from flowyforge.data_plugins.collide_v2 import sample_preparation as prep, streaming
    config = prep.load_sample_config(config_path)
    output = Path(config["output_dir"])
    timing = {}
    if legacy:
        # Instrument the old functions without changing their implementation or inputs.
        def timed(module, name, key):
            original = getattr(module, name)
            def measured(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    timing[key] = timing.get(key, 0.) + time.perf_counter() - start
            setattr(module, name, measured)
        for name, key in (("plan_sample", "discovery_seconds"), ("_export_file", "export_seconds"),
                          ("_validate_export", "parquet_validation_seconds"), ("sha256_file", "file_sha256_seconds")):
            timed(prep, name, key)
        timed(streaming, "sha256_file", "file_sha256_seconds")
        digest = streaming._json_digest
        def fingerprint(value):
            start = time.perf_counter()
            result = digest(value)
            if isinstance(value, list):
                timing["event_fingerprinting_seconds"] = timing.get("event_fingerprinting_seconds", 0.) + time.perf_counter() - start
            return result
        streaming._json_digest = fingerprint
    started = time.perf_counter()
    manifest = prep.prepare_sample(config)
    if legacy:
        phase = time.perf_counter()
        partitions = streaming.create_partition_manifest({"root": str(output)}, seed=config["selection"]["seed"],
            validation_fraction=.2, test_fraction=.2, read_batch_size=2048)
        streaming.save_partition_manifest(partitions, output / "partitions.json")
        timing["partition_generation_seconds"] = time.perf_counter() - phase
        performance = timing
    else:
        performance = manifest["performance"]
        partitions = streaming.load_partition_manifest(output / "partitions.json")
    total = time.perf_counter() - started
    membership = sorted((p["name"], p["label"], e["id"], e["split"]) for p in partitions["processes"] for e in p["events"])
    # All cases must pass this exact scientific membership checksum, independently of policy SHA.
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(membership).encode()).hexdigest()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    result = {"total_wall_seconds": total, "events": len(membership), "events_per_second": len(membership) / total,
              "membership_sha256": fingerprint, "partition_sha256": partitions["sha256"],
              "parent_peak_rss_bytes": peak, "performance": performance}
    (output / "benchmark_case.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Selected-file local preparation YAML (HF inputs must already be downloaded).")
    parser.add_argument("--output-dir", help="New directory for case configs, samples, logs, JSON/CSV/Markdown results.")
    parser.add_argument("--workers", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each policy/worker case; report each observation, no extrapolation.")
    parser.add_argument("--legacy-revision", help="Optional trusted git revision of the pre-refactor code; serial prepare then strict partitions.")
    parser.add_argument("--_case", help=argparse.SUPPRESS)
    parser.add_argument("--_legacy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._case:
        _child(args._case, args._legacy)
        return
    if not args.config or not args.output_dir or args.repeats < 1:
        parser.error("--config, --output-dir and positive --repeats required")
    import yaml
    from flowyforge.data_plugins.collide_v2.sample_preparation import load_sample_config
    from flowyforge.data_plugins.collide_v2.streaming import _resolve_processes
    config = load_sample_config(args.config)
    if config["source"]["kind"] != "local":
        parser.error("Benchmark expects local readable files, to separate network download time.")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = _resolve_processes(config["source"])
    base = {"source": {"kind": "local", "processes": {n: list(map(str, p)) for n, p in paths.items()}},
            "model_config": config["model_config"], "selection": config["selection"],
            "reader": {k: v for k, v in config["readers"]["FullReco"].items() if k != "view"}}
    repo = Path(__file__).resolve().parents[1]
    reference = None
    if args.legacy_revision:
        reference = output / "reference"
        reference.mkdir()
        archive = subprocess.check_output(["git", "archive", "--format=tar", args.legacy_revision], cwd=repo)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # The revision is explicitly supplied by the caller; refuse escaping archive paths.
            for member in tar.getmembers():
                if member.issym() or member.islnk() or not (reference / member.name).resolve().is_relative_to(reference):
                    raise ValueError("Unsupported link/escaping path in reference archive")
            tar.extractall(reference, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    cases = ([("legacy", 1)] if reference else []) + [(p, w) for p in ("fast", "strict") for w in args.workers]
    rows, membership = [], None
    for repeat in range(1, args.repeats + 1):
        for policy, workers in cases:
            tag = f"{policy}-{workers}-{repeat}"
            case_output = output / tag
            raw = {**base, "output_dir": str(case_output)}
            if policy != "legacy":
                raw["partitions"] = {**config["partitions"], "enabled": True,
                                     "check_integrity_SHA256": policy, "num_workers": workers}
            elif any(config["partitions"][k] != .2 for k in ("validation_fraction", "test_fraction")):
                raise ValueError("Legacy benchmark reference uses the original default 60/20/20 splits.")
            path = output / f"{tag}.yaml"
            path.write_text(yaml.safe_dump(raw))
            env = {**os.environ, "PYTHONPATH": str((reference if policy == "legacy" else repo) / "src"),
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            command = [sys.executable, str(Path(__file__).resolve()), "--_case", str(path)]
            if policy == "legacy":
                command.append("--_legacy")
            print(f"Benchmark {tag}: {case_output}", flush=True)
            with (output / f"{tag}.out").open("w") as stdout, (output / f"{tag}.err").open("w") as stderr:
                subprocess.run(command, env=env, cwd=repo, stdout=stdout, stderr=stderr, check=True)
            result = json.loads((case_output / "benchmark_case.json").read_text())
            if membership is not None and result["membership_sha256"] != membership:
                raise ValueError("Benchmark scientific memberships differ; results cannot be compared")
            membership = result["membership_sha256"]
            performance = result["performance"]
            row = {"policy": policy, "workers": workers, "repeat": repeat,
                   **{k: result[k] for k in ("total_wall_seconds", "events_per_second", "parent_peak_rss_bytes")},
                   "discovery_seconds": performance.get("discovery_seconds", 0),
                   "partition_generation_seconds": performance.get("partition_generation_seconds", 0)}
            keys = ("source_verification_seconds", "export_seconds", "partition_collection_seconds",
                    "event_fingerprinting_seconds", "parquet_validation_seconds", "output_sha256_seconds", "file_sha256_seconds")
            for key in keys:
                row[key] = performance.get(key, sum(f.get(key, 0) for f in performance.get("file_timings", [])))
            if policy == "legacy":
                row["source_verification_seconds"] = None  # Not isolated; captured in legacy file_sha256_seconds.
            workers_rss = performance.get("workers", {})
            # Parent overlaps its single-worker process. Multi-worker peaks are NOT simultaneous.
            row["max_worker_peak_rss_bytes"] = max(workers_rss.values(), default=0)
            row["rss_upper_bound_bytes"] = result["parent_peak_rss_bytes"] + (sum(workers_rss.values()) if workers > 1 else 0)
            rows.append(row)
    for row in rows:
        serial = next((r for r in rows if r["policy"] == row["policy"] and r["workers"] == 1 and r["repeat"] == row["repeat"]), None)
        legacy = next((r for r in rows if r["policy"] == "legacy" and r["repeat"] == row["repeat"]), None)
        row["parallel_speedup"] = serial["total_wall_seconds"] / row["total_wall_seconds"] if serial else None
        row["legacy_speedup"] = legacy["total_wall_seconds"] / row["total_wall_seconds"] if legacy else None
    metadata = {"platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(), "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "working_tree": subprocess.check_output(["git", "diff", "--stat"], cwd=repo, text=True),
        "legacy_revision": args.legacy_revision, "input_files": sum(map(len, paths.values())),
        "input_bytes": sum(p.stat().st_size for files in paths.values() for p in files),
        "events_per_process": config["selection"]["events_per_process"], "selection": config["selection"],
        "membership_sha256": membership, "rows": rows,
        "limits": "Local cache order uncontrolled; one fresh process/case; phase sums are worker elapsed times, include nested subphases; RSS lifetime peaks, summed only as a conservative upper bound, not measured concurrent peak. No training or scientific metrics."}
    (output / "benchmark.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Preparation benchmark", "", f"Platform: {metadata['platform']}; CPU: {metadata['cpu']}; logical CPUs: {metadata['logical_cpus']}.",
             f"Input: {metadata['input_files']} files / {metadata['input_bytes']:,} bytes; {metadata['events_per_process']} selected events/process.",
             f"Legacy revision: `{args.legacy_revision}`. Membership checksum: `{membership}`.", "", metadata["limits"], "",
             "| Policy | Workers | Repeat | Wall s | Events/s | Parallel speedup | Legacy speedup | Parent peak MiB | Worker max MiB | RSS upper bound MiB |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        ratio = lambda value: f"{value:.2f}×" if value is not None else "n/a"
        lines.append(f"| {r['policy']} | {r['workers']} | {r['repeat']} | {r['total_wall_seconds']:.3f} | {r['events_per_second']:.1f} | {ratio(r['parallel_speedup'])} | {ratio(r['legacy_speedup'])} | {r['parent_peak_rss_bytes']/2**20:.1f} | {r['max_worker_peak_rss_bytes']/2**20:.1f} | {r['rss_upper_bound_bytes']/2**20:.1f} |")
    lines += ["", "Worker phase totals (seconds; fingerprint time is included in partition collection/generation):", "",
              "| Policy | Workers | Discovery | Source verification | Export | Partition collection | Assignment/save | Event fingerprints | Validation | Output SHA | Legacy file SHA |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        keys = ("discovery_seconds", "source_verification_seconds", "export_seconds", "partition_collection_seconds", "partition_generation_seconds",
                "event_fingerprinting_seconds", "parquet_validation_seconds", "output_sha256_seconds", "file_sha256_seconds")
        lines.append(f"| {r['policy']} | {r['workers']} | " + " | ".join(f"{r[k]:.3f}" if r[k] is not None else "n/a" for k in keys) + " |")
    (output / "benchmark.md").write_text("\n".join(lines) + "\n")
    print(f"Results: {output / 'benchmark.md'}")


if __name__ == "__main__":
    main()
