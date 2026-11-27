#!/bin/bash
#SBATCH --job-name=casmi-09-mlp-test
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
# Scores Thomas's fingerprint MLP on the held-out TEST set (job 07), once. No training: it reruns his
# notebook as job 04 did, where every run now loads the weights job 04 saved, with section 15 added
# at the end (LukeD/bluepebble/mlp_test_cell.md and .py) to score the test set:
#   - the final model (section 12) under his protocol: per molecule, ±10 ppm candidates, class 3 withheld;
#   - the group-split model (section 13), the one compared with DreaMS and MS2DeepScore: per spectrum and
#     per molecule, among the training molecules and among training + test molecules.
# Each model must first reproduce job 04's validation results, or the notebook stops before scoring.
# The rerun writes into a copy of job 04's results folder on work storage, so job 04's committed
# files stay as they were; the test results go to ThomasF/results_bluepebble/mlp_test/, and the executed
# notebook to ThomasF/03a-fingerprint-mlp-bluepebble-100pct-test.ipynb.
# Loading and section 10's timing take most of the time: about 30-60 minutes on one GPU.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
[[ -f "$TEST_DIR/test.parquet" ]] || { echo "!! no test set in $TEST_DIR: run job 07 first (bash submit_all.sh test)"; exit 1; }
use_env casmi

reference="$REPO/ThomasF/results_bluepebble/mlp"                 # job 04's results: read, never written
[[ -f "$reference/final/summary.json" ]] || { echo "!! job 04's results are missing from $reference"; exit 1; }
export MLP_RUN_SIZE="${MLP_RUN_SIZE:-full}"                      # the same plan as job 04, so every run loads
export MLP_CACHE_DIR="$CASMI_WORK/mlp-cache"                     # job 04's fingerprints and weights
export GROUP_SPLIT_DIR="$SPLITS_DIR/stage_3_100pct"
export MLP_RESULTS_DIR="$CASMI_WORK/mlp-test-rerun/mlp"         # a copy of job 04's results for the rerun to write in
export MLP_REFERENCE_DIR="$reference"
export MLP_TEST_OUT="$REPO/ThomasF/results_bluepebble/mlp_test"  # small files, meant to be committed
export TEST_DIR
mkdir -p "$MLP_RESULTS_DIR"
rsync -a --delete "$reference/" "$MLP_RESULTS_DIR/"

copy="$REPO/ThomasF/03a-fingerprint-mlp-bluepebble-100pct-test.ipynb"
python "$BP_DIR/nbparam.py" "$MLP_NOTEBOOK" "$copy"
python "$BP_DIR/nbappend.py" "$copy" "$BP_DIR/mlp_test_cell.md" "$BP_DIR/mlp_test_cell.py"
echo "== executing $copy"
start=$SECONDS
python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$copy" \
    || { echo "!! the MLP was NOT scored on the test set: see the error above, or open $copy"; exit 1; }
echo "== finished in $(( (SECONDS - start) / 60 )) min; MLP test results in ThomasF/results_bluepebble/mlp_test/"
