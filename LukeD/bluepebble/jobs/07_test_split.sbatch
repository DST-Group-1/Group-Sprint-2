#!/bin/bash
#SBATCH --job-name=casmi-07-test-split
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Builds the held-out TEST set that every trained model is scored on once, at the end. Michael's
# splits have no test set, so it is the test half of LukeD/01's held-out molecules (heldout.parquet,
# split == "test"): 2,998 molecules in classes 1-3, 13,769 spectra on the full data.
# Before writing anything it checks the 100% split the models trained on: no test spectrum, no
# spectrum of a class 2 or 3 test molecule, and no timsTOF spectrum of a class 1 test molecule may be
# in its training or validation files. If any is, the job stops and nothing should be scored.
# Writes $TEST_DIR: test.parquet (the same columns as Michael's validation.parquet),
# test_molecules.parquet (class, pool, and whether each molecule has training spectra) and
# manifest.json, which is also copied to LukeD/results/test_heldout/ to commit. CPU only, a minute or two.
# Never train on that folder or use it to choose an epoch or a setting: validation is for that.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env casmi

args=(--derived "$DERIVED_DIR" --split "$SPLITS_DIR/stage_3_100pct" --out "$TEST_DIR")
if [[ -f "$RAW_DIR/train.parquet" ]]; then
    args+=(--original "$RAW_DIR/train.parquet")      # exact source_part values from its row groups
else
    echo "!! $RAW_DIR/train.parquet not found, so source_part is inferred from the spectra_train part files"
fi
python "$BP_DIR/make_test_split.py" "${args[@]}" || { echo "!! test set NOT built: see above"; exit 1; }

mkdir -p "$REPO/LukeD/results/test_heldout"
cp "$TEST_DIR/manifest.json" "$REPO/LukeD/results/test_heldout/manifest.json"
echo "== test set written to $TEST_DIR"
