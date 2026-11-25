#!/bin/bash
# After the jobs finish: write what Slurm recorded about each one (wall time, peak memory, what was
# allocated, how it ended) to logs/sacct.csv, the resource table for the report, and list the small
# result files that are meant to be committed.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
ids=$(cut -f3 "$LOG_DIR/submitted.tsv" | paste -sd, -)
sacct -j "$ids" --parsable2 \
    --format=JobID,JobName%30,Partition,State,ExitCode,Start,Elapsed,TotalCPU,MaxRSS,ReqMem,AllocTRES%60,NodeList \
    > "$LOG_DIR/sacct.csv"
column -s '|' -t "$LOG_DIR/sacct.csv" | grep -v -E '\.(extern|batch)' || true

cd "$REPO"
echo
echo "== results to commit (all small):"
git status --short --untracked-files=all -- \
    LukeD ThomasF/results_bluepebble MichaelM DavidA/results_bluepebble \
    | grep -v -E '\.(npy|pt|ckpt)$' || true
