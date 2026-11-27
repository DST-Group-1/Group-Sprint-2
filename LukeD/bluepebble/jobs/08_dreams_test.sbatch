#!/bin/bash
#SBATCH --job-name=casmi-08-dreams-test
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Scores the fine-tuned DreaMS from job 06 on the held-out TEST set (job 07), once. No training: it
# loads the best epoch job 06 saved ($CHECKPOINT_DIR/dreams_100pct.pt) into Michael's exact model.
#   1. Re-predicts the validation spectra and checks they match job 06's validation_predictions.csv.
#      If they don't, the model was not rebuilt faithfully and it stops before touching the test set.
#   2. Scores the test spectra; small files go to MichaelM/results_bluepebble/dreams_100pct_3epochs_test/.
# About 10-15 minutes on one GPU, most of it loading and preprocessing.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
[[ -f "$TEST_DIR/test.parquet" ]] || { echo "!! no test set in $TEST_DIR: run job 07 first (bash submit_all.sh test)"; exit 1; }
use_env dreams

run="dreams_100pct_${DREAMS_MAX_EPOCHS}epochs"
python "$BP_DIR/dreams_score_test.py" \
    --checkpoint "$CHECKPOINT_DIR/dreams_100pct.pt" \
    --ssl-checkpoint "$MODELS_DIR/ssl_model.ckpt" \
    --split "$SPLITS_DIR/stage_3_100pct" \
    --test "$TEST_DIR" \
    --reference-predictions "$REPO/MichaelM/validation_results_bluepebble/$run/validation_predictions.csv" \
    --out "$REPO/MichaelM/results_bluepebble/${run}_test" \
    || { echo "!! DreaMS was NOT scored on the test set: see above"; exit 1; }
echo "== DreaMS test results in MichaelM/results_bluepebble/${run}_test/"
