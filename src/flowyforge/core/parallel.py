"""Bounded, ordered file jobs; no scheduler or external runtime dependency."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import sys


def peak_rss_bytes():
    """Process lifetime RSS high-water mark (not a sum or simultaneous job peak)."""
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * (1 if sys.platform == "darwin" else 1024))
    except ImportError:
        return None


def _init_file_worker():
    import pyarrow as pa
    import torch
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    torch.set_num_threads(1)


def map_files(function, jobs, num_workers):
    """Yield in input order, with at most N in-flight jobs/results.

    Spawn avoids inheriting Arrow/Torch thread pools. On failure, cancel queued
    work and join running writers BEFORE callers clean up their owned outputs.
    """
    if type(num_workers) is not int or num_workers not in (1, 2, 4, 8):
        raise ValueError("num_workers must be one of 1, 2, 4, 8.")
    if num_workers == 1:
        for job in jobs:
            yield function(job)
        return
    pool = ProcessPoolExecutor(num_workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_init_file_worker)
    pending, iterator = deque(), iter(jobs)
    try:
        for _ in range(num_workers):
            if (job := next(iterator, None)) is not None:
                pending.append(pool.submit(function, job))
        while pending:
            yield pending.popleft().result()
            if (job := next(iterator, None)) is not None:
                pending.append(pool.submit(function, job))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
