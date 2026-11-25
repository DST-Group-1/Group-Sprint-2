#!/bin/bash
#SBATCH --job-name=casmi-06-dreams
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
# Michael's fine-tuned DreaMS notebook on the 100% group split, with his hyperparameters.
# Training stops after DREAMS_MAX_EPOCHS (config.sh, default 3). On the RTX 4080 an epoch took 567 s
# for the 9% split's 179,108 spectra, so an epoch of the 2.27 million here is about 2 hours there and
# several hours on the RTX 2080 used here (Michael trains in fp32), hence the 48 h limit. The log's
# first epoch time shows how long the rest will take. 3 epochs are 6.8 million training examples, more than the 3.6 million of the 9%
# run's 20 epochs, so the model has had more updates than the converged 9% run.
# Weights (over a GB) go to work storage; the small results go to MichaelM/results_bluepebble/.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env dreams

stage=stage_3_100pct
out="$REPO/MichaelM/results_bluepebble/dreams_100pct_${DREAMS_MAX_EPOCHS}epochs"
mkdir -p "$out" "$CHECKPOINT_DIR"

run_nb "$DREAMS_NOTEBOOK" DreaMS_tuned-bluepebble-100pct \
    "DATA_DIR=Path('$SPLITS_DIR/$stage')" \
    "DREAMS_CHECKPOINT=Path('$MODELS_DIR/ssl_model.ckpt')" \
    "OUTPUT_DIR=Path('$out')" \
    "EXPERIMENT_2_OUTPUT_DIR=Path('$REPO/MichaelM/for_michael_tuned_dreams/results/fine_tuned_dreams_experiment_2')" \
    "EXPERIMENT_2_TRAIN_PATH=Path('$SPLITS_DIR/stage_1_5pct/train.parquet')" \
    "MAX_EPOCHS=$DREAMS_MAX_EPOCHS" \
    "BEST_MODEL_PATH=Path('$CHECKPOINT_DIR/dreams_100pct.pt')"
