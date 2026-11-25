#!/bin/bash
# Settings shared by every BluePebble job in this folder. Edit the lines marked EDIT once.
# submit_all.sh and every job script source this file.

# ---- EDIT: scheduler -----------------------------------------------------------------------------
export HPC_ACCOUNT="CHANGE_ME"   # your project code; list yours with:
                                 #   sacctmgr show user withassoc format=account where user=$USER
export CPU_PARTITION="compute"   # check the names and time limits with: sinfo -s
export GPU_PARTITION="gpu"
export GPU_GRES="gpu:1"          # pin ONE card type for a whole scaling series, so every point of a
                                 # curve ran on the same hardware, e.g. gpu:rtx_3090:1
                                 # (the type names are in: sinfo -o "%20P %40G %l")

# ---- Paths ---------------------------------------------------------------------------------------
# $HOME is small (about 20 GB), so the repo, data, environments and results all live on work storage.
export CASMI_WORK="${CASMI_WORK:-${WORK:-/user/work/$USER}/casmi26}"
export REPO="$CASMI_WORK/Group-Sprint-2"                 # the git clone
export COMPETITION="enveda-CASMI26-molecule-id-mass-spectra"
export RAW_DIR="$REPO/$COMPETITION"                      # symlink to the Kaggle download (setup makes it)
export DERIVED_DIR="$REPO/$COMPETITION-derived"          # written by LukeD/01
export CASMI26_DERIVED_DIR="$DERIVED_DIR"                # read by LukeD/01 and ThomasF/03a
export SPLITS_DIR="$REPO/model_datasets"                 # Michael's splits: stage_1_5pct, stage_2_9pct, stage_3_100pct
export MODELS_DIR="$CASMI_WORK/models"                   # pretrained weights: ssl_model.ckpt, ms2deepscore_model.pt
export CHECKPOINT_DIR="$CASMI_WORK/checkpoints"          # weights trained here (large, never committed)
export OUTPUT_DIR_LARGE="$CASMI_WORK/results"            # large outputs such as embeddings (never committed)
export CONDA_ROOT="$CASMI_WORK/miniforge3"
export KAGGLEHUB_CACHE="$CASMI_WORK/kagglehub"
export BP_DIR="$REPO/LukeD/bluepebble"
export LOG_DIR="$BP_DIR/logs"

# ---- Which notebooks the jobs run --------------------------------------------------------------
export MLP_NOTEBOOK="$REPO/ThomasF/03a-fingerprint-mlp.ipynb"
export DREAMS_NOTEBOOK="$REPO/MichaelM/DreaMS_tuned_scaled_executed.ipynb"
export MS2DS_NOTEBOOK="$REPO/DavidA/MS2DeepScore/MS2DEEPSCORE_5%"
export DREAMS_MAX_EPOCHS="${DREAMS_MAX_EPOCHS:-3}"       # see the README: 20 epochs at 100% is days of GPU time
