"""Lightweight invocation statistics, separate from scientific metrics/checkpoints."""

from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import torch


def _cpu_model():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
                text=True, stderr=subprocess.DEVNULL, timeout=2,
            ).strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unavailable"


def _write_report(output, rows):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "computation_stats.md"
    previous = path.read_text(encoding="utf-8") if path.exists() else "# Computation statistics\n"
    # A resumed run adds one invocation; never discard earlier measurements.
    body = "\n## Invocation " + rows["Started (UTC)"] + "\n\n| Statistic | Value |\n| --- | --- |\n"
    for key, value in rows.items():
        value = f"{value:.6f}" if isinstance(value, float) else str(value)
        body += f"| {key} | {value.replace('|', '/').replace(chr(10), ' ')} |\n"
    body += (
        "\nScope: the selected runner call, including preflight, data loading, training, "
        "evaluation and automatic reports where applicable. Excludes imports/config loading "
        "and writing these statistics. Times describe this invocation only, not earlier epochs "
        "restored with resume. CPU times include all threads of the main process. "
        "Child CPU times count only terminated, waited-for subprocesses (including DataLoader "
        "workers), not live children or other cluster jobs; unavailable on Windows. "
        "User/system times use OS accounting resolution and may round to zero.\n\n"
        "CUDA interval is measured by two events on the selected device's current stream "
        "around the runner call: it includes host/I/O idle gaps and is **not GPU busy/kernel "
        "time or utilization**. GPU active time requires a separate profiler and is not "
        "collected. MPS has no corresponding portable event measurement here. GPU work "
        "is synchronized at the measurement boundaries.\n"
    )
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(previous + body, encoding="utf-8")
    temporary.replace(path)


@contextmanager
def computation_statistics(output, *, device, stage="train", num_workers=0, resume=None):
    """Append statistics after a successful CLI run, without changing training artifacts.

    ``device`` must be the resolved torch device actually used by the runner.
    Failed/rejected runs leave existing reports untouched. No RNG is consumed.
    """
    device = torch.device(device)
    start_event = end_event = None
    gpu_interval = "N/A (CPU run)" if device.type == "cpu" else "N/A (no portable MPS event timer)"
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(device))
        elif device.type == "mps":
            torch.mps.synchronize()
    except RuntimeError as exc:
        start_event = end_event = None
        gpu_interval = f"N/A (GPU timing unavailable: {exc})"
    started = datetime.now(timezone.utc).isoformat()
    wall_start, cpu_start, usage_start = time.perf_counter(), time.process_time(), os.times()
    yield
    # Deliberately not a finally block: invalid resume/input must not touch outputs.
    try:
        if start_event is not None:
            end_event.record(torch.cuda.current_stream(device))
            torch.cuda.synchronize(device)
            gpu_interval = start_event.elapsed_time(end_event) / 1000.0
        elif device.type == "mps":
            torch.mps.synchronize()
    except RuntimeError as exc:
        gpu_interval = f"N/A (GPU timing unavailable: {exc})"
    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    usage = os.times()
    ended = datetime.now(timezone.utc).isoformat()
    child_user = usage.children_user - usage_start.children_user
    child_system = usage.children_system - usage_start.children_system
    rows = {
        "Started (UTC)": started, "Finished (UTC)": ended, "Stage": stage,
        "Resume source": str(resume) if resume else "None (new invocation)",
        "Real / wall time (s)": wall,
        "Main process CPU time (s)": cpu,
        "Main process user CPU time (s)": usage.user - usage_start.user,
        "Main process system CPU time (s)": usage.system - usage_start.system,
        "Subprocess user CPU time (s)": child_user if os.name != "nt" else "N/A",
        "Subprocess system CPU time (s)": child_system if os.name != "nt" else "N/A",
        "Subprocess total CPU time (s)": child_user + child_system if os.name != "nt" else "N/A",
        "CUDA stream elapsed interval (s)": gpu_interval,
        "GPU active/kernel time (s)": "N/A (not profiled)",
        "Device used": str(device), "CPU model": _cpu_model(),
        "Logical CPUs (host)": os.cpu_count(),
        "PyTorch intra-op threads": torch.get_num_threads(),
        "PyTorch inter-op threads": torch.get_num_interop_threads(),
        "DataLoader workers": num_workers,
        "Host": platform.node(), "OS": platform.platform(), "Architecture": platform.machine(),
        "Python": platform.python_version(), "PyTorch": str(torch.__version__), "PID": os.getpid(),
    }
    try:
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            rows.update({"GPU model": properties.name, "GPU memory (GiB)": properties.total_memory / 2**30,
                         "CUDA runtime": torch.version.cuda})
        else:
            rows["GPU model"] = "Apple Metal / MPS (system GPU)" if device.type == "mps" else "None used"
        _write_report(output, rows)
    except (OSError, RuntimeError) as exc:
        # Measurement/reporting failure must not invalidate a completed training run.
        warnings.warn(f"Could not write computation_stats.md: {exc}", RuntimeWarning)
