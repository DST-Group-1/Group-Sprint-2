#!/bin/bash
#SBATCH --job-name=casmi-06-dreams
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --array=0-2
# Michael's fine-tuned DreaMS notebook on each group split, same hyperparameters, one GPU per task.
# Training is capped at DREAMS_MAX_EPOCHS (config.sh, default 3): on the RTX 4080 an epoch took 567 s
# for 179,108 spectra, so one epoch of the 100% split is about 2 hours there and longer on these
# cards, and 20 epochs would be days. Equal epochs on one card type give seconds per epoch against
# training spectra, and accuracy at equal epochs; the converged 5% and 9% runs from the 4080 stay as they are.
# Weights (over a GB at 100%) go to work storage; the small results go to MichaelM/results_bluepebble/.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env dreams

stage="$(stage_name "${SLURM_ARRAY_TASK_ID:-0}")"
out="$REPO/MichaelM/results_bluepebble/dreams_${stage}_${DREAMS_MAX_EPOCHS}epochs"
mkdir -p "$out" "$CHECKPOINT_DIR"

run_nb "$DREAMS_NOTEBOOK" "DreaMS_tuned-bluepebble-$stage" \
    "DATA_DIR=Path('$SPLITS_DIR/$stage')" \
    "DREAMS_CHECKPOINT=Path('$MODELS_DIR/ssl_model.ckpt')" \
    "OUTPUT_DIR=Path('$out')" \
    "EXPERIMENT_2_OUTPUT_DIR=Path('$REPO/MichaelM/for_michael_tuned_dreams/results/fine_tuned_dreams_experiment_2')" \
    "EXPERIMENT_2_TRAIN_PATH=Path('$SPLITS_DIR/stage_1_5pct/train.parquet')" \
    "MAX_EPOCHS=$DREAMS_MAX_EPOCHS" \
    "BEST_MODEL_PATH=Path('$CHECKPOINT_DIR/dreams_$stage.pt')"
