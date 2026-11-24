"""Accounting without profiling overhead or changes to model/RNG state."""

from types import SimpleNamespace

import pytest
import torch

from flowyforge.training import run_statistics as stats


def test_cpu_counters_are_deltas_and_resume_appends(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, "_cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(stats.time, "perf_counter", iter([100., 110., 200., 205.]).__next__)
    monkeypatch.setattr(stats.time, "process_time", iter([30., 32., 35., 36.]).__next__)
    usage = [SimpleNamespace(user=u, system=s, children_user=cu, children_system=cs)
             for u, s, cu, cs in [(20., 10., 4., 2.), (21.5, 10.5, 5., 2.25),
                                  (22., 13., 7., 3.), (22.75, 13.25, 8., 3.5)]]
    monkeypatch.setattr(stats.os, "times", iter(usage).__next__)
    rng = torch.get_rng_state().clone()
    with stats.computation_statistics(tmp_path, device="cpu", num_workers=2):
        pass
    path = tmp_path / "computation_stats.md"
    original = path.read_text()
    for field, value in [("Real / wall time (s)", 10.), ("Main process CPU time (s)", 2.),
                         ("Main process user CPU time (s)", 1.5),
                         ("Main process system CPU time (s)", .5)]:
        assert f"| {field} | {value:.6f} |" in original
    if stats.os.name != "nt":
        assert "| Subprocess total CPU time (s) | 1.250000 |" in original
    else:
        assert "| Subprocess total CPU time (s) | N/A |" in original
    assert "| CPU model | Test CPU |" in original
    assert "| GPU model | None used |" in original
    assert "| DataLoader workers | 2 |" in original
    assert "N/A (CPU run)" in original
    with stats.computation_statistics(tmp_path, device="cpu", resume="last_checkpoint.pt"):
        pass
    assert path.read_text().startswith(original)
    assert path.read_text().count("## Invocation ") == 2
    assert "| Resume source | last_checkpoint.pt |" in path.read_text()
    assert torch.equal(torch.get_rng_state(), rng)


def test_rejected_run_preserves_existing_report(tmp_path):
    path = tmp_path / "computation_stats.md"
    path.write_text("previous run\n")
    with pytest.raises(ValueError, match="incompatible"):
        with stats.computation_statistics(tmp_path, device="cpu"):
            raise ValueError("incompatible checkpoint")
    assert path.read_text() == "previous run\n"
    assert list(tmp_path.iterdir()) == [path]


def test_cuda_interval_is_seconds_and_not_claimed_as_active_time(tmp_path, monkeypatch):
    calls = []
    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self, stream):
            calls.append(("record", stream))

        def elapsed_time(self, other):
            assert isinstance(other, Event)
            return 2500.  # PyTorch CUDA events return milliseconds.

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: str(device))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", str(device))))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(name="Test GPU", total_memory=8 * 2**30))
    monkeypatch.setattr(stats, "_cpu_model", lambda: "Test CPU")
    with stats.computation_statistics(tmp_path, device="cuda:1"):
        pass
    content = (tmp_path / "computation_stats.md").read_text()
    assert "| CUDA stream elapsed interval (s) | 2.500000 |" in content
    assert "| GPU active/kernel time (s) | N/A (not profiled) |" in content
    assert "| GPU model | Test GPU |" in content
    assert "| GPU memory (GiB) | 8.000000 |" in content
    assert calls == [("sync", "cuda:1"), ("record", "cuda:1"), ("record", "cuda:1"), ("sync", "cuda:1")]


def test_mps_synchronizes_but_does_not_invent_gpu_time(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("sync"))
    with stats.computation_statistics(tmp_path, device="mps"):
        pass
    content = (tmp_path / "computation_stats.md").read_text()
    assert "N/A (no portable MPS event timer)" in content
    assert calls == ["sync", "sync"]


def test_report_io_error_does_not_fail_completed_training(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("read-only report")
    monkeypatch.setattr(stats, "_write_report", fail)
    with pytest.warns(RuntimeWarning, match="Could not write computation_stats.md"):
        with stats.computation_statistics(tmp_path, device="cpu"):
            pass
