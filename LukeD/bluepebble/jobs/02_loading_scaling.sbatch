#!/bin/bash
#SBATCH --job-name=casmi-02-scaling
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --mem=0
# LukeD/02 on a whole compute node. --exclusive matters for two reasons: no other job's load shows up
# in the timings, and the notebook sizes itself with os.cpu_count() and psutil's total memory, which
# report the whole node; on a shared node those would overstate what the job may use.
# The worker counts go past the laptop's 8 (the notebook drops any above the node's core count).
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
use_env casmi

run_nb "$REPO/LukeD/02-loading-and-scaling.ipynb" 02-loading-and-scaling-bluepebble \
    "WORKER_COUNTS=[1, 2, 4, 8, 12, 16, 24, 32]" \
    'SPARK_DRIVER_MEMORY="8g"' \
    'RESULTS_DIR=HERE / "benchmark_results" / "bluepebble"'
