#!/bin/bash
#SBATCH --job-name=casmi-05-ms2ds
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
# David's MS2DeepScore notebook in its "100pct" mode: the pretrained model (no training) embeds every
# cleaned training spectrum outside the validation set as references, then searches them. It scores
# the same validation spectra as his 5% run (the 5% split's known molecules, 10,153 spectra), so the
# two compare directly. The 9% run took 3 minutes on the RTX 4080, so this is about 40 minutes.
# Embeddings (about 4.6 GB) stay on work storage; the small results go to DavidA/results_bluepebble/100pct/.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env ms2ds

out="$OUTPUT_DIR_LARGE/ms2deepscore_100pct"
mkdir -p "$out"
run_nb "$MS2DS_NOTEBOOK" MS2DEEPSCORE-bluepebble-100pct \
    "MODE='100pct'" \
    "DATA_DIR=Path('$SPLITS_DIR/stage_1_5pct')" \
    "FULL_GLOB=r'$DERIVED_DIR/spectra_train/*.parquet'" \
    "MODEL_FILE=Path('$MODELS_DIR/ms2deepscore_model.pt')" \
    "OUTPUT_DIR=Path('$out')"

mkdir -p "$REPO/DavidA/results_bluepebble/100pct"
rsync -a --exclude='*.npy' "$out/" "$REPO/DavidA/results_bluepebble/100pct/"
