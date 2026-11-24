#!/bin/bash
# Run on the login node after submission: bash scripts/slurm/status.sh 12345
# Also accepts a single array element, e.g. 12345_1. This script never modifies jobs.
set -euo pipefail
job=${1:-}
if [[ $# != 1 || ! $job =~ ^[0-9]+(_[0-9]+)?$ ]]; then
    echo "Usage: bash scripts/slurm/status.sh JOB_ID[_ARRAY_INDEX]" >&2
    exit 2
fi
# Each query gets its own text snapshot; old accounting/logs are kept.
log_dir=${FLOWYFORGE_STATUS_DIR:-outputs/slurm_logs}
mkdir -p "$log_dir"
report="$log_dir/status-${job}-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
query() {
    printf '\n$ '; printf '%q ' "$@"; printf '\n'
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "UNAVAILABLE: $1; query from a Slurm login node. Job outcome UNKNOWN."
    elif ! "$@"; then
        echo "Query unavailable/record expired/not yet accounted; this is NOT evidence of job success."
    fi
}
{
    printf 'Job %s; queried UTC %s\n' "$job" "$(date -u +%FT%TZ)"
    # Pending/running array elements plus pending reason; finished jobs disappear.
    query squeue --array --jobs="$job" --format='%.22i %.12P %.24j %.12T %.12M %R'
    # Includes .batch/.extern and srun steps: do NOT use sacct -X, which hides
    # step-level memory/CPU statistics. Missing values depend on site accounting.
    query sacct --jobs="$job" --format=JobID%24,State%24,ExitCode,Elapsed,AllocCPUS,TotalCPU,MaxRSS,NodeList%30
    # Live usage (may be empty for an array parent; query JOBID_INDEX if needed).
    query sstat --allsteps --jobs="$job" --format=JobID,AveCPU,AveRSS,MaxRSS
    # Shows paths to stdout/stderr, requested resources and scheduling reason.
    query scontrol show job "$job"
    cat <<'TEXT'

Interpretation: COMPLETED with ExitCode=0:0 on the job AND its training step
indicates scheduler success. Inspect .out/.err, metrics.json and report artifacts
to verify the scientific run. PENDING/RUNNING is not a final outcome.
FAILED, OUT_OF_MEMORY, TIMEOUT, CANCELLED, NODE_FAIL or PREEMPTED need inspection.
An empty queue or missing accounting record does NOT mean success; sacct may lag.
CPUTime would be allocated CPUs * elapsed; TotalCPU above is accounted CPU usage.
GPU activity is not inferred from CPU usage; GPU accounting is site-dependent.
TEXT
} 2>&1 | tee "$report"
printf '\nSaved status snapshot: %s\n' "$report"
