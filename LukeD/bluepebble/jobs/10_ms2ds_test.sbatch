#!/bin/bash
#SBATCH --job-name=casmi-10-ms2ds-test
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
# Scores MS2DeepScore on the held-out TEST set (job 07), once. Nothing is trained or re-embedded: it
# reuses the pretrained model and the reference library job 05 embedded (every spectrum of the 100%
# training split, in $OUTPUT_DIR_LARGE/ms2deepscore_100pct), with David's own preprocessing and search.
#   1. Repeats job 05's validation search and checks it gives job 05's saved answers, or stops.
#   2. Searches the test spectra; small files go to DavidA/results_bluepebble/100pct_test/.
# About 10 minutes on one GPU, most of it moving the 4.6 GB of reference embeddings onto the card.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
[[ -f "$TEST_DIR/test.parquet" ]] || { echo "!! no test set in $TEST_DIR: run job 07 first (bash submit_all.sh test)"; exit 1; }
use_env ms2ds

python "$BP_DIR/ms2ds_score_test.py" \
    --model "$MODELS_DIR/ms2deepscore_model.pt" \
    --reference-dir "$OUTPUT_DIR_LARGE/ms2deepscore_100pct" \
    --check-data-dir "$SPLITS_DIR/stage_1_5pct" \
    --check-predictions "$REPO/DavidA/results_bluepebble/100pct/100pct_predictions.csv" \
    --test "$TEST_DIR" \
    --out "$REPO/DavidA/results_bluepebble/100pct_test" \
    || { echo "!! MS2DeepScore was NOT scored on the test set: see above"; exit 1; }
echo "== MS2DeepScore test results in DavidA/results_bluepebble/100pct_test/"
