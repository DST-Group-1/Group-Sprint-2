#!/bin/bash
# Settings shared by every BluePebble job in this folder. Edit the lines marked EDIT once.
# submit_all.sh and every job script source this file.

# ---- Scheduler (from sacctmgr and sinfo on BluePebble, 25 Nov 2026) ---------------------------------
export HPC_ACCOUNT="math002941"  # sacctmgr show user withassoc format=account where user=$USER
# A comma-separated list lets Slurm start a job in whichever partition frees up first.
export CPU_PARTITION="short,compute"   # short: 3 days, compute: 14 days; every CPU job here is under 6 h
export GPU_PARTITION="gpu"             # 14 days: for the jobs longer than 6 h (the MLP, DreaMS at 100%)
export GPU_SHORT_PARTITION="gpu_short,gpu"   # gpu_short: 6 h, RTX 2080 and 3090 only: for the short GPU jobs
# One card type for every GPU job, so all the 100% runs used the same hardware. RTX 2080 (11 GB):
# BluePebble's only RTX 3090 node (8 cards) was fully booked until Thursday night on 25 Nov, while
# most 2080 nodes had free cards. DreaMS peaked at 4.4 GB on Michael's 4080 run, so 11 GB is enough.
# The 2080 is in both gpu and gpu_short. Avoid the V100s: they are labelled both V100 and v100, and
# recent torch builds no longer support them.
export GPU_GRES="gpu:rtx_2080:1"       # other choices: gpu:rtx_3090:1 (24 GB, one node), gpu:a100:1 (gpu only)

# ---- Paths ---------------------------------------------------------------------------------------
# $HOME is small (about 20 GB), so the repo, data, environments and results all live on work storage.
export CASMI_WORK="${CASMI_WORK:-${WORK:-/user/work/$USER}/casmi26}"
export REPO="$CASMI_WORK/Group-Sprint-2"                 # the git clone
export COMPETITION="enveda-CASMI26-molecule-id-mass-spectra"
export RAW_DIR="$REPO/$COMPETITION"                      # symlink to the Kaggle download (setup makes it)
export DERIVED_DIR="$REPO/$COMPETITION-derived"          # written by LukeD/01
export CASMI26_DERIVED_DIR="$DERIVED_DIR"                # read by LukeD/01 and ThomasF/03a
export SPLITS_DIR="$REPO/model_datasets"                 # Michael's splits: stage_1_5pct (defines the validation set), stage_3_100pct
export MODELS_DIR="$CASMI_WORK/models"                   # pretrained weights: ssl_model.ckpt, ms2deepscore_model.pt
export CHECKPOINT_DIR="$CASMI_WORK/checkpoints"          # weights trained here (large, never committed)
export OUTPUT_DIR_LARGE="$CASMI_WORK/results"            # large outputs such as embeddings (never committed)
export CONDA_ROOT="$CASMI_WORK/miniforge3"
export KAGGLEHUB_CACHE="$CASMI_WORK/kagglehub"
export PIP_CACHE_DIR="$CASMI_WORK/pip-cache"             # torch wheels would otherwise fill the 25 GB home quota
export BP_DIR="$REPO/LukeD/bluepebble"
export LOG_DIR="$BP_DIR/logs"

# ---- Which notebooks the jobs run --------------------------------------------------------------
export MLP_NOTEBOOK="$REPO/ThomasF/03a-fingerprint-mlp.ipynb"
export DREAMS_NOTEBOOK="$REPO/MichaelM/DreaMS_tuned_scaled_executed.ipynb"
export MS2DS_NOTEBOOK="$REPO/DavidA/MS2DeepScore/MS2DEEPSCORE_5%"
export DREAMS_MAX_EPOCHS="${DREAMS_MAX_EPOCHS:-3}"       # see the README: 20 epochs at 100% is days of GPU time
