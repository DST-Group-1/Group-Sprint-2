#!/bin/bash
#SBATCH --job-name=casmi-01-data
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Regenerates the cleaned data from the Kaggle download by running LukeD/01 unchanged, then checks
# that it matches the laptop run count for count (the split is hash-based, so it should).
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env casmi

run_nb "$REPO/LukeD/01-data-and-validation.ipynb" 01-data-and-validation-bluepebble

python - <<EOF
import json, sys
counts = json.load(open("$DERIVED_DIR/manifest.json"))["counts"]
# From the manifest of the laptop run (16 Nov 2026).
expected = {"spectra in": 2539608, "training spectra out": 2296272, "query spectra out": 27539, "peaks out": 71076838}
bad = {k: (counts.get(k), v) for k, v in expected.items() if counts.get(k) != v}
for k, v in expected.items():
    print(f"{k:>22}: {counts.get(k):>12,}  (laptop: {v:,})")
print("cleaning took", round(counts["seconds"], 1), "s here")
sys.exit("!! the cleaned data differs from the laptop run: " + str(bad) if bad else 0)
EOF
echo "== cleaned data matches the laptop run"
