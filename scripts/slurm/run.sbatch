#!/bin/bash
# FlowyForge on TERABIT/HPC Bubble: ONE independent run per array element.
# Submit from the repository root; create outputs/slurm_logs BEFORE sbatch.
# CNAF settings supplied by the user (login: ui-terabit.cr.cnaf.infn.it).
# Override account/QOS/partition on the sbatch command line for another project/site.
# CLI sbatch options override these conservative defaults. No GPU is requested
# here: for CUDA add --gres=gpu:1 and pass "cuda" as the fourth script argument.
# Human-readable name shown by squeue and substituted into log filenames (%x).
#SBATCH --job-name=flowyforge
# Project allocation and scheduling QoS; these are not the SSH login username.
#SBATCH --account=kamiknows
#SBATCH --qos=kamiknows
# CPU default matches the no-GPU resource default below. Preferred ML GPU runs:
# sbatch --partition=bare-metal-GPU --gres=gpu:1 ... run.sbatch ... cuda
# CNAF also provides all-nodes (site default); choose an explicit partition here.
#SBATCH --partition=bare-metal-nodes
# Keep one model on one node; the current trainer has no MPI/DDP support.
#SBATCH --nodes=1
# One task means one Python trainer; extra tasks would NOT speed up this model.
#SBATCH --ntasks=1
# CPUs shared by Torch threads and data-loading workers within that trainer.
#SBATCH --cpus-per-task=6
# Host RAM per node, independent of GPU VRAM; a limit, not measured usage.
#SBATCH --mem=16G
# Wall-clock limit, including input/output and final evaluation/report creation.
#SBATCH --time=24:00:00
# # SBATCH --time=INFINITE
# One FullReco run by default. --array=0-1%2 runs both views concurrently.
#SBATCH --array=0%1
# %x = job name, %A = array job ID, %a = array index; no collisions between jobs.
#SBATCH --output=outputs/slurm_logs/%x-%A_%a.out
#SBATCH --error=outputs/slurm_logs/%x-%A_%a.err
# Keep logs if a site administrator requeues; never implicitly restart weights.
#SBATCH --open-mode=append
#SBATCH --no-requeue

# Fail on missing variables/failed commands, including commands in pipelines.
set -euo pipefail
if (( $# < 4 )); then
    echo "Usage: sbatch [Slurm options] scripts/slurm/run.sbatch classification|pretraining CONFIG OUTPUT_ROOT cpu|cuda [--python PATH] [runner options]" >&2
    exit 2
fi
: "${SLURM_JOB_ID:?Submit with sbatch; do not train on the login node}"
: "${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
runner=$1; config=$2; output_root=$3; device=$4
shift 4
case "$runner" in classification|pretraining) ;; *) echo "Invalid runner: $runner" >&2; exit 2;; esac
case "$device" in cpu|cuda) ;; *) echo "Device must be cpu or cuda (no silent fallback)." >&2; exit 2;; esac
if [[ ${SLURM_NTASKS:-1} != 1 || ${SLURM_JOB_NUM_NODES:-1} != 1 ]]; then
    echo "This runner requires one task on one node; parallelize independent runs with arrays." >&2
    exit 2
fi

# Array membership changes ONLY the view/output directory, never event splits.
# A single L1T run uses --array=1%1. Higher indices are deliberately rejected.
case "${SLURM_ARRAY_TASK_ID:-0}" in
    0) view=FullReco; view_dir=fullreco;;
    1) view=L1T; view_dir=l1t;;
    *) echo "Only array indices 0 (FullReco) and 1 (L1T) are supported." >&2; exit 2;;
esac
# Slurm copies this batch file into its spool directory; use the submission
# directory, NOT dirname "$0", to locate the checkout and relative data paths.
cd "${FLOWYFORGE_REPO:-$SLURM_SUBMIT_DIR}"
[[ -f scripts/run_classification.py ]] || { echo "Submit from the FlowyForge root." >&2; exit 2; }
output="$output_root/$view_dir"
job_tag="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$output/slurm/$job_tag"
job_log="$output/slurm/$job_tag/run.txt"
started=$SECONDS

# Record wrapper completion even on Python/setup failure. SIGKILL, node failure
# and some timeouts cannot run this trap: sacct is the authoritative job status.
finish() {
    rc=$?
    trap - EXIT
    printf 'finished_utc=%s\nexit_code=%s\nwrapper_wall_seconds=%s\n' \
        "$(date -u +%FT%TZ)" "$rc" "$((SECONDS - started))" | tee -a "$job_log"
    exit "$rc"  # Preserve failure for Slurm; never hide it behind tee/echo.
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
printf 'started_utc=%s\njob=%s\nnode=%s\nrunner=%s\nview=%s\ndevice=%s\noutput=%s\n' \
    "$(date -u +%FT%TZ)" "$job_tag" "$(hostname)" "$runner" "$view" "$device" "$output" | tee -a "$job_log"

# Optional trusted site setup: e.g. module load <site-python> <site-cuda>.
# No module names are guessed and no packages/downloads are installed in a job.
if [[ -n ${FLOWYFORGE_SETUP:-} ]]; then source "$FLOWYFORGE_SETUP"; fi
python_bin=${FLOWYFORGE_PYTHON:-"$PWD/.venv/bin/python"}

# Pass supported options as Bash array elements (no eval, safe paths with spaces).
# View/device/output/resume are controlled here to prevent array jobs colliding.
# Options use TWO tokens: --epochs 20, not --epochs=20. {view} expands to the
# lower-case output suffix in encoder paths, allowing separate pretrained views.
extra=(); worker_override=""; stage=train
while (( $# )); do
    (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
    option=$1; value=$2; shift 2
    case "$option" in
        # Wrapper-only override: never forward this option to either Python CLI.
        --python) python_bin=$value; continue;;
        --epochs|--batch-size|--seed|--partition-manifest|--data-root|--mode|--model-config|--process|--process-name|--max-events-per-class|--downstream-mode) ;;
        --num-workers) worker_override=$value;;
        --stage) stage=$value; [[ $stage == train || $stage == inspect ]] || { echo "Only train/inspect are supported." >&2; exit 2; };;
        --encoder-checkpoint) value=${value//\{view\}/$view_dir};;
        *) echo "Unsupported/managed option: $option; see README Slurm section." >&2; exit 2;;
    esac
    extra+=("$option" "$value")
done

# Resolve precedence before validation: CLI > environment > repository venv.
# Log even an invalid selection, so setup failures identify the attempted path.
printf 'python=%s\n' "$python_bin" | tee -a "$job_log"
[[ -x "$python_bin" ]] || { echo "Python is not executable: $python_bin. Use --python PATH or FLOWYFORGE_PYTHON." >&2; exit 2; }

# Respect training.cpu_threads from the EXISTING YAML; OMP alone cannot override
# torch.set_num_threads in the runner. Reject an undersized allocation. Worker
# count is a loading parallelism control, not a number of training processes.
settings=$("$python_bin" - "$config" "$worker_override" "$SLURM_CPUS_PER_TASK" "$device" <<'PY'
import sys
import yaml
import torch
with open(sys.argv[1], encoding="utf-8") as stream:
    training = (yaml.safe_load(stream) or {}).get("training", {})
threads = training.get("cpu_threads", 4)
workers = int(sys.argv[2]) if sys.argv[2] else training.get("num_workers", 0)
if type(threads) is not int or type(workers) is not int or threads < 1 or workers < 0:
    raise SystemExit("cpu_threads must be positive and num_workers non-negative integers.")
if threads + workers > int(sys.argv[3]):
    raise SystemExit("Request --cpus-per-task >= training.cpu_threads + num_workers (or reduce the YAML/worker settings).")
if sys.argv[4] == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
    raise SystemExit("CUDA run requires exactly one visible GPU: request --gres=gpu:1 and a CUDA-enabled PyTorch build.")
print(threads, workers)
PY
)
read -r threads workers <<< "$settings"
# Bound native thread pools; the runner controls Torch threads using the YAML.
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
# Never change CUDA_VISIBLE_DEVICES: Slurm assigns/remaps the allocated GPU.
command=("$python_bin" -u "scripts/run_${runner}.py" --config "$config"
         --stage "$stage" --view "$view" --output-dir "$output" --device "$device"
         ${extra[@]+"${extra[@]}"})  # Empty-array expansion also works with Bash 3 + nounset.

# Explicit resume only; never silently restart or automatically skip a run.
# Keep the SAME OUTPUT_ROOT, config, device and resources when resubmitting.
case "${FLOWYFORGE_RESUME:-0}" in
    0) ;;
    1)
        [[ $stage == train ]] || { echo "Resume requires stage train." >&2; exit 2; }
        checkpoint=last_checkpoint.pt
        [[ $runner == pretraining ]] && checkpoint=checkpoint.pt
        [[ -f "$output/$checkpoint" ]] || { echo "Missing resume checkpoint: $output/$checkpoint" >&2; exit 2; }
        command+=(--resume "$output/$checkpoint");;
    *) echo "FLOWYFORGE_RESUME must be 0 or 1." >&2; exit 2;;
esac
{
    printf 'cpus_allocated=%s\ntorch_threads=%s\nloader_workers=%s\n' "$SLURM_CPUS_PER_TASK" "$threads" "$workers"
    printf 'command='; printf '%q ' "${command[@]}"; printf '\n'
    "$python_bin" --version
    git rev-parse HEAD 2>/dev/null || true
} | tee -a "$job_log"

# One Slurm step, one Python trainer, using the allocated CPUs. Stdout/stderr
# flow directly into the .out/.err files above. srun propagates Python failure.
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores "${command[@]}"
