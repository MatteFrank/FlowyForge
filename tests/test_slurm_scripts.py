"""Exercise batch dispatch with fake Slurm, including real tiny CPU runners."""

import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest
import torch
import yaml

from test_masked_pretraining import make_run_config

REPO = Path(__file__).resolve().parents[1]
BATCH = REPO / "scripts/slurm/run.sbatch"
STATUS = REPO / "scripts/slurm/status.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="Slurm scripts require Bash")


@pytest.fixture
def cluster(tmp_path):
    config, _ = make_run_config(tmp_path)
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    # Record the exact argv that srun would launch. Optionally execute the real
    # CPU CLI, without requesting cluster resources or launching other workers.
    srun = bin_dir / "srun"
    srun.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
args = sys.argv[4:]
with open(os.environ["CAPTURE"], "w") as f:
    json.dump({"args": args, "omp": os.environ["OMP_NUM_THREADS"]}, f)
if os.environ.get("EXECUTE") == "1":
    os.execv(args[0], args)
sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
''')
    srun.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "SLURM_JOB_ID": "123", "SLURM_ARRAY_JOB_ID": "123", "SLURM_ARRAY_TASK_ID": "0",
           "SLURM_CPUS_PER_TASK": "6", "SLURM_NTASKS": "1", "SLURM_JOB_NUM_NODES": "1",
           "SLURM_SUBMIT_DIR": str(REPO), "FLOWYFORGE_REPO": str(REPO),
           "FLOWYFORGE_PYTHON": sys.executable, "FLOWYFORGE_RESUME": "0",
           "FLOWYFORGE_SETUP": "", "CAPTURE": str(tmp_path / "argv.json")}
    return config, tmp_path / "outputs with spaces", env, bin_dir


def launch(cluster, runner="classification", extra=(), device="cpu", **updates):
    config, output, env, _ = cluster
    return subprocess.run(["bash", str(BATCH), runner, str(config), str(output), device, *extra],
                          env={**env, **updates}, text=True, capture_output=True)


@pytest.mark.parametrize("index,view", [("0", "FullReco"), ("1", "L1T")])
def test_array_dispatch_and_literal_paths(cluster, index, view):
    result = launch(cluster, extra=["--encoder-checkpoint", "outputs/enc/{view}/encoder.pt", "--num-workers", "2"],
                    SLURM_ARRAY_TASK_ID=index)
    assert result.returncode == 0, result.stderr
    _, output, env, _ = cluster
    saved = json.loads(Path(env["CAPTURE"]).read_text())
    args = saved["args"]
    assert args[args.index("--view") + 1] == view
    assert args[args.index("--output-dir") + 1] == str(output / view.lower())
    assert args[args.index("--encoder-checkpoint") + 1] == f"outputs/enc/{view.lower()}/encoder.pt"
    assert saved["omp"] == "1"
    assert "exit_code=0" in (output / view.lower() / f"slurm/123_{index}/run.txt").read_text()


def test_defaults_without_extra_arguments(cluster):
    result = launch(cluster)
    assert result.returncode == 0, result.stderr
    assert "unbound variable" not in result.stderr
    args = json.loads(Path(cluster[2]["CAPTURE"]).read_text())["args"]
    assert args[-2:] == ["--device", "cpu"]


@pytest.mark.parametrize("selection", ["cli", "environment", "default"])
@pytest.mark.parametrize("runner,index,view", [("classification", "0", "FullReco"), ("pretraining", "1", "L1T")])
def test_python_selection_precedence_and_all_invocations(cluster, tmp_path, selection, runner, index, view):
    _, output, env, _ = cluster
    # Isolate the default venv from the developer's real checkout/environment.
    repo = tmp_path / "checkout with spaces"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts/run_classification.py").touch()
    env.update(FLOWYFORGE_REPO=str(repo), SLURM_ARRAY_TASK_ID=index)
    env.pop("FLOWYFORGE_PYTHON")
    selected = repo / ".venv/bin/python" if selection == "default" else tmp_path / f"{selection} venv/bin/python"
    selected.parent.mkdir(parents=True)
    calls = tmp_path / "python calls.txt"
    selected.write_text("#!/bin/bash\n" +
                        f"printf '%s\\n' \"$1\" >> {shlex.quote(str(calls))}\n" +
                        f"exec {shlex.quote(sys.executable)} \"$@\"\n")
    selected.chmod(0o755)
    extra = ["--epochs", "2"]
    if selection == "cli":
        # An invalid environment default must not be checked before the override.
        env["FLOWYFORGE_PYTHON"] = str(tmp_path / "missing environment python")
        extra += ["--python", str(selected)]
    elif selection == "environment":
        env["FLOWYFORGE_PYTHON"] = str(selected)
    extra += ["--num-workers", "0"]
    result = launch(cluster, runner, extra)
    assert result.returncode == 0, result.stdout + result.stderr
    args = json.loads(Path(env["CAPTURE"]).read_text())["args"]
    assert args[:3] == [str(selected), "-u", f"scripts/run_{runner}.py"]
    assert "--python" not in args
    assert args[args.index("--epochs") + 1] == "2"
    assert args[args.index("--num-workers") + 1] == "0"
    assert args[args.index("--view") + 1] == view
    assert args[args.index("--output-dir") + 1] == str(output / view.lower())
    assert calls.read_text().splitlines() == ["-", "--version"]  # Preflight and version use the same interpreter.
    log = (output / view.lower() / f"slurm/123_{index}/run.txt").read_text()
    assert f"python={selected}\n" in log and f"python={selected}\n" in result.stdout


@pytest.mark.parametrize("kind", ["missing_value", "missing_file", "not_executable", "empty"])
def test_invalid_python_override_never_starts_training(cluster, tmp_path, kind):
    selected = tmp_path / "invalid python"
    if kind == "not_executable":
        selected.write_text("#!/bin/bash\nexit 0\n")
        selected.chmod(0o644)
    extra = ["--python"] if kind == "missing_value" else ["--python", "" if kind == "empty" else str(selected)]
    result = launch(cluster, extra=extra)
    assert result.returncode == 2
    assert ("Missing value for --python" if kind == "missing_value" else "Python is not executable") in result.stderr
    assert not Path(cluster[2]["CAPTURE"]).exists()


@pytest.mark.parametrize("runner,checkpoint", [("classification", "last_checkpoint.pt"), ("pretraining", "checkpoint.pt")])
def test_explicit_resume_and_failure_exit_preserved(cluster, runner, checkpoint):
    _, output, env, _ = cluster
    directory = output / "fullreco"
    directory.mkdir(parents=True)
    (directory / checkpoint).write_text("dispatch test only")
    result = launch(cluster, runner, ["--epochs", "20"], FLOWYFORGE_RESUME="1", FAKE_EXIT="7")
    assert result.returncode == 7
    args = json.loads(Path(env["CAPTURE"]).read_text())["args"]
    assert args[-2:] == ["--resume", str(directory / checkpoint)]
    assert "exit_code=7" in (directory / "slurm/123_0/run.txt").read_text()


@pytest.mark.parametrize("extra,env,expected", [
    (["--num-workers", "6"], {}, "cpus-per-task"),
    ([], {"SLURM_ARRAY_TASK_ID": "2"}, "Only array indices"),
    ([], {"SLURM_NTASKS": "2"}, "one task"),
    ([], {"FLOWYFORGE_RESUME": "1"}, "Missing resume checkpoint"),
    (["--output-dir", "collision"], {}, "managed option"),
    (["--stage", "evaluate"], {}, "Only train/inspect"),
])
def test_invalid_dispatch_never_starts_training(cluster, extra, env, expected):
    result = launch(cluster, extra=extra, **env)
    assert result.returncode != 0
    assert expected in result.stderr
    assert not Path(cluster[2]["CAPTURE"]).exists()


@pytest.mark.parametrize("runner,stage,index", [
    ("classification", "train", "0"), ("pretraining", "inspect", "1"), ("pretraining", "train", "1"),
])
def test_real_cpu_runner_through_batch_wrapper(cluster, runner, stage, index):
    config, output, _, _ = cluster
    if runner == "classification":
        raw = yaml.safe_load(config.read_text())
        raw.pop("pretraining")
        config.write_text(yaml.safe_dump(raw))
    result = launch(cluster, runner, ["--stage", stage, "--python", sys.executable], EXECUTE="1",
                    SLURM_ARRAY_TASK_ID=index, FLOWYFORGE_PYTHON="/missing/default/python")
    assert result.returncode == 0, result.stdout + result.stderr
    directory = output / ("fullreco" if index == "0" else "l1t")
    assert (directory / "computation_stats.md").exists()
    assert (directory / ("metrics.json" if stage == "train" else "preflight.json")).exists()


@pytest.mark.skipif(torch.cuda.is_available(), reason="Requires CPU-only environment to test missing CUDA")
def test_gpu_request_cannot_silently_fall_back_to_cpu(cluster):
    result = launch(cluster, device="cuda")
    assert result.returncode != 0
    assert "exactly one visible GPU" in result.stderr
    assert not Path(cluster[2]["CAPTURE"]).exists()


def test_status_saves_queries_and_does_not_hide_failed_job(cluster, tmp_path):
    _, _, env, bin_dir = cluster
    for name in ("squeue", "sacct", "sstat", "scontrol"):
        path = bin_dir / name
        path.write_text("#!/bin/bash\nprintf '%s\\n' '123_1 TIMEOUT 0:15'\n")
        path.chmod(0o755)
    result = subprocess.run(["bash", str(STATUS), "123_1"], capture_output=True, text=True,
                            env={**env, "FLOWYFORGE_STATUS_DIR": str(tmp_path / "logs")})
    assert result.returncode == 0  # Status query succeeded; NOT a training-success exit code.
    report, = (tmp_path / "logs").glob("status-*.txt")
    content = report.read_text()
    assert "TIMEOUT 0:15" in content and "NOT mean success" in content
    assert "sacct" in content and "sstat" in content and "MaxRSS" in content
    assert "-X" not in content
