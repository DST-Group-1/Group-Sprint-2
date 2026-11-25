#!/bin/bash
# shellcheck disable=SC2046  # $(after ...) is meant to vanish when there is no dependency
# Submit the pipeline with its dependencies, from the login node:
#
#   bash submit_all.sh                       # everything
#   bash submit_all.sh smoke                 # or chosen steps: smoke data scaling splits mlp ms2ds dreams
#   bash submit_all.sh mlp dreams            # e.g. rerun two steps once data and splits exist
#
# A step waits (afterok) only for steps submitted in the same call, so a failed job stops everything
# that depends on it. Job ids are appended to logs/submitted.tsv for collect.sh.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
if [[ "$HPC_ACCOUNT" == CHANGE_ME ]]; then
    echo "Set HPC_ACCOUNT in config.sh first (sacctmgr show user withassoc format=account where user=\$USER)"
    exit 1
fi
mkdir -p "$LOG_DIR"
STEPS=" ${*:-smoke data scaling splits mlp ms2ds dreams} "
cpu=(--account="$HPC_ACCOUNT" --partition="$CPU_PARTITION")
gpu=(--account="$HPC_ACCOUNT" --partition="$GPU_PARTITION" --gres="$GPU_GRES")
declare -A ID

after() {  # after <step> ... -> a --dependency option for the steps submitted in this call
    local ids=()
    for s in "$@"; do [[ -n "${ID[$s]:-}" ]] && ids+=("${ID[$s]}"); done
    if (( ${#ids[@]} )); then local IFS=:; echo "--dependency=afterok:${ids[*]}"; fi
}

submit() {  # submit <step> <job script> <sbatch options...>
    local step="$1" script="$2"; shift 2
    [[ "$STEPS" == *" $step "* ]] || return 0
    local log="$LOG_DIR/%x-%j.out"
    grep -q -- "--array" "$script" && log="$LOG_DIR/%x-%A_%a.out"
    ID[$step]=$(sbatch --parsable --output="$log" "$@" "$script")
    printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$step" "${ID[$step]}" "$GPU_GRES" >> "$LOG_DIR/submitted.tsv"
    echo "submitted $step as job ${ID[$step]}"
}

submit smoke   jobs/00_smoke_gpu.sbatch       "${gpu[@]}"
submit data    jobs/01_data.sbatch            "${cpu[@]}"
submit scaling jobs/02_loading_scaling.sbatch "${cpu[@]}" $(after data)
submit splits  jobs/03_group_splits.sbatch    "${cpu[@]}" $(after data)
submit mlp     jobs/04_mlp.sbatch             "${gpu[@]}" $(after smoke splits)
submit ms2ds   jobs/05_ms2deepscore.sbatch    "${gpu[@]}" $(after smoke splits)
submit dreams  jobs/06_dreams.sbatch          "${gpu[@]}" $(after smoke splits)
echo "watch with: squeue --me    logs in: $LOG_DIR"
