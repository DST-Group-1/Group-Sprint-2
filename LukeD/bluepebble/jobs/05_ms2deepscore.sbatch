#!/bin/bash
#SBATCH --job-name=casmi-05-ms2ds
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --array=0-2
# David's MS2DeepScore notebook (pretrained model, no training: it embeds the reference spectra and
# searches them). One array task per reference set; they can run side by side on different GPUs.
#   task 0: references = stage_1_5pct training spectra   (his "5pct" mode)
#   task 1: references = stage_2_9pct training spectra   (the same mode on the 9% split)
#   task 2: references = every cleaned training spectrum (his "100pct" mode, validation rows excluded)
# Tasks 0 and 2 score the same validation spectra (the 5% split's known molecules), so they compare
# directly. Embeddings (about 4.6 GB at 100%) stay on work storage; the small results are copied into
# DavidA/results_bluepebble/.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env ms2ds

task="${SLURM_ARRAY_TASK_ID:-0}"
case "$task" in
    0) mode=5pct;   data=stage_1_5pct; label=5pct ;;
    1) mode=5pct;   data=stage_2_9pct; label=9pct ;;
    2) mode=100pct; data=stage_1_5pct; label=100pct ;;
esac
out="$OUTPUT_DIR_LARGE/ms2deepscore_$label"
mkdir -p "$out"

run_nb "$MS2DS_NOTEBOOK" "MS2DEEPSCORE-bluepebble-$label" \
    "MODE='$mode'" \
    "DATA_DIR=Path('$SPLITS_DIR/$data')" \
    "FULL_GLOB=r'$DERIVED_DIR/spectra_train/*.parquet'" \
    "MODEL_FILE=Path('$MODELS_DIR/ms2deepscore_model.pt')" \
    "OUTPUT_DIR=Path('$out')"

mkdir -p "$REPO/DavidA/results_bluepebble/$label"
rsync -a --exclude='*.npy' "$out/" "$REPO/DavidA/results_bluepebble/$label/"
