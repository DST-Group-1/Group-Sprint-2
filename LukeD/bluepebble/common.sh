#!/bin/bash
# Helpers for the job scripts. Sourced after config.sh.

set -euo pipefail

use_env() {  # use_env <conda env name>
    # Some activation scripts (openjdk's, for one) read unset variables, so -u is off while they run.
    set +u
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$1"
    set -u
    echo "== env $1: $(python --version 2>&1) at $(command -v python)"
}

match_threads_to_allocation() {
    # os.cpu_count() reports the whole node, not what Slurm gave this job, so libraries that size
    # their thread pools from it oversubscribe a shared node. Pin them to the allocation instead.
    local n="${SLURM_CPUS_PER_TASK:-$(nproc)}"
    export OMP_NUM_THREADS="$n" MKL_NUM_THREADS="$n" OPENBLAS_NUM_THREADS="$n" POLARS_MAX_THREADS="$n"
}

log_hardware() {
    echo "== job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+ task $SLURM_ARRAY_TASK_ID} on $(hostname), $(date -Is)"
    echo "== cores: allocated ${SLURM_CPUS_PER_TASK:-?}, usable $(nproc), on the node $(nproc --all); memory: ${SLURM_MEM_PER_NODE:-?} MB requested"
    lscpu | grep -E "^(Model name|Socket\(s\)|Core\(s\) per socket|Thread\(s\) per core)" || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
    fi
}

run_nb() {  # run_nb <notebook> <name of the copy, without .ipynb> [NAME=EXPRESSION ...]
    # Writes a copy next to the original with the given settings changed, then executes the copy
    # in place, so the original stays untouched and the copy keeps this run's outputs.
    local notebook="$1" name="$2"; shift 2
    local copy; copy="$(dirname "$notebook")/$name.ipynb"
    python "$BP_DIR/nbparam.py" "$notebook" "$copy" "$@"
    echo "== executing $copy"
    local start=$SECONDS
    python -m jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$copy"
    echo "== finished $copy in $(( (SECONDS - start) / 60 )) min"
}

stage_name() {  # stage_name <array index> -> Michael's split folder name
    local stages=(stage_1_5pct stage_2_9pct stage_3_100pct)
    echo "${stages[$1]}"
}
