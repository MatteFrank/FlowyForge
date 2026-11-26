#!/usr/bin/env python
"""Benchmark preparation + partitions, or strict partitions on existing inputs.

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


# Optional strict-partition microbenchmark. These wrappers exist only in this
# process/its spawned workers; production fingerprint semantics are untouched.
_PROFILE = {}


def _add(key, elapsed):
    _PROFILE[key] = _PROFILE.get(key, 0.) + elapsed


class _TableProxy:
    def __init__(self, table):
        self.table = table

    def __getattr__(self, name):
        return getattr(self.table, name)

    def __len__(self):
        return len(self.table)

    def to_pylist(self):
        started = time.perf_counter()
        result = self.table.to_pylist()
        _add("arrow_to_python_seconds", time.perf_counter() - started)
        return result


class _ArrowProxy:
    def __init__(self, arrow):
        self.arrow = arrow
        class Table:
            @staticmethod
            def from_batches(*args, **kwargs):
                started = time.perf_counter()
                # Unwrap instrumented RecordBatches without copying buffers.
                batches, *rest = args
                result = arrow.Table.from_batches([getattr(b, 'table', b) for b in batches], *rest, **kwargs)
                _add("recordbatch_to_table_seconds", time.perf_counter() - started)
                return _TableProxy(result)
        self.Table = Table

    def __getattr__(self, name):
        return getattr(self.arrow, name)


class _ParquetProxy:
    def __init__(self, reader):
        self.reader = reader

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.reader.close()

    def __getattr__(self, name):
        return getattr(self.reader, name)

    def iter_batches(self, *args, **kwargs):
        batches = iter(self.reader.iter_batches(*args, **kwargs))
        while True:
            started = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                _add("parquet_read_decode_batch_seconds", time.perf_counter() - started)
                return
            _add("parquet_read_decode_batch_seconds", time.perf_counter() - started)
            _add("arrow_batch_bytes", batch.nbytes)
            yield _TableProxy(batch)


def _instrument_strict():
    import hashlib
    from types import SimpleNamespace
    from flowyforge.data_plugins.collide_v2 import streaming as s
    original_sha, original_digest, original_parquet = s.sha256_file, s._json_digest, s.pq.ParquetFile
    def sha(path):
        started = time.perf_counter()
        result = original_sha(path)
        _add("file_sha256_seconds", time.perf_counter() - started)
        return result
    def digest(value):
        event = isinstance(value, list) and len(value) == 2 and isinstance(value[1], dict)
        started = time.perf_counter()
        if event:
            # Identical canonical encoder; separate its CPU cost from cryptography.
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=s._json_value).encode()
            _add("event_json_utf8_seconds", time.perf_counter() - started)
            _add("event_json_bytes", len(payload))
            started = time.perf_counter()
            result = hashlib.sha256(payload).hexdigest()
            _add("event_sha256_seconds", time.perf_counter() - started)
        else:
            result = original_digest(value)
            key = "schema_seconds" if isinstance(value, dict) and "fields" in value else "manifest_json_sha_seconds"
            _add(key, time.perf_counter() - started)
        return result
    s.sha256_file, s._json_digest = sha, digest
    s.pa = _ArrowProxy(s.pa)
    s.pq = SimpleNamespace(ParquetFile=lambda *a, **kw: _ParquetProxy(original_parquet(*a, **kw)))
    if hasattr(s, '_arrow_rows'):
        rows = s._arrow_rows
        depth = 0
        def converted(table, **kwargs):
            # Time iterator advancement, excluding JSON/SHA done by the consumer.
            nonlocal depth
            iterator = iter(rows(table, **kwargs))
            while True:
                started = time.perf_counter()
                depth += 1
                try:
                    value = next(iterator)
                except StopIteration:
                    return
                finally:
                    depth -= 1
                    if depth == 0:
                        _add("arrow_to_python_seconds", time.perf_counter() - started)
                yield value
        s._arrow_rows = converted
    if hasattr(s, '_schema_digest'):
        schema = s._schema_digest
        def schema_digest(value):
            started = time.perf_counter()
            old = _PROFILE.get('schema_seconds', 0.)
            result = schema(value)
            _PROFILE['schema_seconds'] = old + time.perf_counter() - started
            return result
        s._schema_digest = schema_digest
    validate = s.validate_partition_manifest
    def checked(value):
        started = time.perf_counter()
        result = validate(value)
        _add('manifest_validation_seconds', time.perf_counter() - started)
        return result
    s.validate_partition_manifest = checked


def _profile_worker():
    from flowyforge.core.parallel import _init_file_worker
    _init_file_worker()
    _instrument_strict()


def _profile_file(job):
    from flowyforge.core.parallel import peak_rss_bytes
    from flowyforge.data_plugins.collide_v2.streaming import _scan_partition_file
    _PROFILE.clear()
    result = _scan_partition_file(job)
    return result, dict(_PROFILE), peak_rss_bytes(), os.getpid()


def _partition_child(config_path):
    import inspect
    import resource
    from flowyforge.data_plugins.collide_v2 import streaming as s
    config = json.loads(Path(config_path).read_text())
    _instrument_strict()
    worker_peaks = {}
    if 'num_workers' in inspect.signature(s.create_partition_manifest).parameters:
        from flowyforge.core import parallel
        parallel._init_file_worker = _profile_worker
        def profiled_map(function, jobs, workers):
            started = time.perf_counter()
            iterator = parallel.map_files(function if workers == 1 else _profile_file, jobs, workers)
            for item in iterator:
                if workers == 1:
                    result = item
                else:
                    result, profile, peak, pid = item
                    for key, value in profile.items():
                        _add(key, value)
                    worker_peaks[str(pid)] = max(worker_peaks.get(str(pid), 0), peak or 0)
                consumer = time.perf_counter()
                yield result
                _add('parent_inventory_merge_seconds', time.perf_counter() - consumer)
            _add('file_pipeline_wall_seconds', time.perf_counter() - started)
        s.map_files = profiled_map
    scan = s._scan_partition_source
    def scanned(*a, **kw):
        started = time.perf_counter()
        result = scan(*a, **kw)
        elapsed = time.perf_counter() - started
        _add('source_scan_wall_seconds', elapsed)
        if 'file_pipeline_wall_seconds' in _PROFILE:
            _add('inventory_sort_duplicate_seconds', elapsed - _PROFILE['file_pipeline_wall_seconds']
                 + _PROFILE.get('parent_inventory_merge_seconds', 0.))
        return result
    s._scan_partition_source = scanned
    output = Path(config['output_dir']); output.mkdir()
    kwargs = {'seed': 42, 'read_batch_size': config['batch_size']}
    if 'num_workers' in inspect.signature(s.create_partition_manifest).parameters:
        kwargs['num_workers'] = config['workers']
    elif config['workers'] != 1:
        raise ValueError('Original reference has no process pool; use one worker.')
    started, cpu = time.perf_counter(), time.process_time()
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    manifest = s.create_partition_manifest({'root': config['source']}, **kwargs)
    create_wall = time.perf_counter() - started
    phase = time.perf_counter()
    s.save_partition_manifest(manifest, output / 'partitions.json')
    save_wall = time.perf_counter() - phase
    elapsed = time.perf_counter() - started
    child_end = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = time.process_time() - cpu + child_end.ru_utime + child_end.ru_stime - children.ru_utime - children.ru_stime
    parent_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    records = sum(p['num_rows'] for p in manifest['processes'])
    result = {**config, 'wall_seconds': elapsed, 'cpu_seconds': cpu, 'events': records,
              'events_per_second': records / elapsed, 'input_bytes': sum(f['size_bytes'] for p in manifest['processes'] for f in p['files']),
              'parent_peak_rss_bytes': parent_peak, 'worker_peak_rss_bytes': worker_peaks,
              'rss_upper_bound_bytes': parent_peak + sum(worker_peaks.values()),
              'partition_sha256': manifest['sha256'], 'profile': {**_PROFILE, 'manifest_save_seconds': save_wall,
              'partition_assembly_seconds': create_wall - _PROFILE['source_scan_wall_seconds']}}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


def _partition_benchmark(args):
    import statistics
    repo = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=False)
    variants = []
    for label, revision in [('original', args.legacy_revision), ('task1', args.task1_revision)]:
        if not revision:
            continue
        reference = output / (label + '_reference'); reference.mkdir()
        archive = subprocess.check_output(['git', 'archive', '--format=tar', revision], cwd=repo)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk() or not (reference / member.name).resolve().is_relative_to(reference):
                    raise ValueError('Unsafe reference archive path')
            tar.extractall(reference, **({'filter': 'data'} if hasattr(tarfile, 'data_filter') else {}))
        variants.append((label, reference))
    variants.append(('optimized', repo))
    results, expected = [], None
    for repeat in range(1, args.repeats + 1):
        for label, reference in variants:
            for workers in ([1] if label == 'original' else args.workers):
                tag = f'{label}-{workers}-{repeat}'
                config = {'source': str(Path(args.partition_source).resolve()), 'batch_size': args.batch_size,
                          'workers': workers, 'variant': label, 'repeat': repeat, 'output_dir': str(output / tag)}
                path = output / (tag + '.json'); path.write_text(json.dumps(config))
                env = {**os.environ, 'PYTHONPATH': str(reference / 'src'),
                       'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
                print(f'Strict partition benchmark {tag}', flush=True)
                with (output / (tag + '.out')).open('w') as out, (output / (tag + '.err')).open('w') as err:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--_partition-case', str(path)],
                                   env=env, cwd=repo, stdout=out, stderr=err, check=True)
                result = json.loads((output / tag / 'result.json').read_text())
                if expected is not None and result['partition_sha256'] != expected:
                    raise ValueError('Strict fingerprints/manifest differ from reference; benchmark rejected.')
                expected = result['partition_sha256']; results.append(result)
    metadata = {'platform': platform.platform(), 'python': platform.python_version(), 'cpus': os.cpu_count(),
                'reference_original': args.legacy_revision, 'reference_task1': args.task1_revision,
                'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
                'results': results, 'scope': 'Fresh process per case; instrumented warm/cache-order-dependent runs. Phase times are inclusive worker sums, not additive wall time. Read/decode/RecordBatch creation are one Arrow native iterator step; to_pylist includes nested Python allocation. RSS sum is an upper bound, not concurrent peak. Original supports serial only.'}
    (output / 'strict_benchmark.json').write_text(json.dumps(metadata, indent=2) + '\n')
    reference_workers = 1 if args.legacy_revision or 1 in args.workers else min(args.workers)
    reference_label = 'original' if args.legacy_revision else 'optimized'
    reference = statistics.median(r['wall_seconds'] for r in results if r['variant'] == reference_label and r['workers'] == reference_workers)
    lines = ['# Strict partition benchmark', '', metadata['scope'], '',
        f'| Variant | Workers | Repeat | Wall s | CPU s | Events/s | Speedup vs {reference_label}/{reference_workers} | Parent RSS MiB | RSS upper bound MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    flat = []
    for r in results:
        lines.append(f"| {r['variant']} | {r['workers']} | {r['repeat']} | {r['wall_seconds']:.3f} | {r['cpu_seconds']:.3f} | {r['events_per_second']:.1f} | {reference/r['wall_seconds']:.2f} | {r['parent_peak_rss_bytes']/2**20:.1f} | {r['rss_upper_bound_bytes']/2**20:.1f} |")
        flat.append({k: v for k, v in r.items() if k not in ('profile', 'worker_peak_rss_bytes')} | r['profile'])
    keys = ('file_sha256_seconds', 'parquet_read_decode_batch_seconds', 'recordbatch_to_table_seconds', 'arrow_to_python_seconds',
            'event_json_utf8_seconds', 'event_sha256_seconds', 'schema_seconds', 'inventory_sort_duplicate_seconds', 'partition_assembly_seconds',
            'manifest_validation_seconds', 'manifest_save_seconds')
    lines += ['', '| Variant | Workers | ' + ' | '.join(keys) + ' |', '| --- | ---: | ' + ' | '.join(['---:'] * len(keys)) + ' |']
    for r in results:
        lines.append(f"| {r['variant']} | {r['workers']} | " + ' | '.join(f"{r['profile'][k]:.4f}" if k in r['profile'] else 'n/a' for k in keys) + ' |')
    (output / 'strict_benchmark.md').write_text('\n'.join(lines) + '\n')
    with (output / 'strict_benchmark.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader(); writer.writerows(flat)
    print(f"Results: {output / 'strict_benchmark.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-source", help="Benchmark strict partition creation only, on this existing process-directory root.")
    parser.add_argument("--task1-revision", help="Task-1 git reference for --partition-source (e.g. c336c07).")
    parser.add_argument("--batch-size", type=int, default=64, help="Arrow rows/batch for --partition-source; bounded Python expansion in optimized code.")
    parser.add_argument("--_partition-case", help=argparse.SUPPRESS)
    parser.add_argument("--config", help="Selected-file local preparation YAML (HF inputs must already be downloaded).")
    parser.add_argument("--output-dir", help="New directory for case configs, samples, logs, JSON/CSV/Markdown results.")
    parser.add_argument("--workers", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each policy/worker case; report each observation, no extrapolation.")
    parser.add_argument("--legacy-revision", help="Optional trusted git revision of the pre-refactor code; serial prepare then strict partitions.")
    parser.add_argument("--_case", help=argparse.SUPPRESS)
    parser.add_argument("--_legacy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._partition_case:
        _partition_child(args._partition_case)
        return
    if args.partition_source:
        if not args.output_dir or args.repeats < 1 or args.batch_size < 1:
            parser.error("--output-dir and positive repeats/batch-size required")
        _partition_benchmark(args)
        return
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
