#!/bin/bash
#SBATCH --job-name=casmi-04-mlp
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
# Thomas's fingerprint MLP on the 100% group split (his section 13), scored on the same validation
# spectra as DreaMS and MS2DeepScore.
# His notebook already holds a 100% model under his own protocol (results_9pct/mlp/final: 2,289,876
# spectra, 19 min on the RTX 4080). To avoid redoing sections 1-12, this job works on a copy of his
# saved runs: a saved run is found by a hash of its settings, so every run is loaded rather than
# retrained, except the few whose weights were never committed (the final model, about 20 min). Its
# section 10 also measures this card, so compute.json ends up with the same model timed on the 4080
# and the RTX 2080. If the saved runs did not load for some reason, the notebook retrains its whole
# "full" plan, which takes several hours more on a 2080 but should still fit in the time asked for.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env casmi

export MLP_RUN_SIZE=full
export MLP_RESULTS_DIR="$REPO/ThomasF/results_bluepebble/mlp"   # small files, meant to be committed
export MLP_CACHE_DIR="$CASMI_WORK/mlp-cache"                     # fingerprints and weights, large
export GROUP_SPLIT_DIR="$SPLITS_DIR/stage_3_100pct"
if [[ ! -d "$MLP_RESULTS_DIR" ]]; then
    mkdir -p "$(dirname "$MLP_RESULTS_DIR")"
    cp -r "$REPO/ThomasF/results_9pct/mlp" "$MLP_RESULTS_DIR"   # the original folder is left as it is
fi
run_nb "$MLP_NOTEBOOK" 03a-fingerprint-mlp-bluepebble-100pct
