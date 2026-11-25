#!/bin/bash
#SBATCH --job-name=casmi-04-mlp
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
# Thomas's fingerprint MLP notebook, once per group split. The first pass trains the "full" plan of
# sections 1-12 (1% to 100% of the protocol subsets, the ablations and the width grid; his notebook
# puts this at 2-5 hours on one GPU) and saves every run. The later passes load those saved runs and
# only train section 13 on the next split. The passes run one after another, not as an array, because
# they share the saved-run folder.
# Results go to a fresh folder, so every point of the curves is retrained on this one card type rather
# than mixed with the RTX 4080 runs already in results_9pct.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env casmi

export MLP_RUN_SIZE=full
export MLP_RESULTS_DIR="$REPO/ThomasF/results_bluepebble/mlp"   # small files, meant to be committed
export MLP_CACHE_DIR="$CASMI_WORK/mlp-cache"                     # fingerprints and weights, large
for stage in stage_1_5pct stage_2_9pct stage_3_100pct; do
    export GROUP_SPLIT_DIR="$SPLITS_DIR/$stage"
    run_nb "$MLP_NOTEBOOK" "03a-fingerprint-mlp-bluepebble-$stage"
done
