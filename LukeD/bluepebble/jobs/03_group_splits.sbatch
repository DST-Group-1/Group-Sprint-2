#!/bin/bash
#SBATCH --job-name=casmi-03-splits
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
# Rebuilds Michael's shared splits from the cleaned data with his own notebooks, plus a third stage
# that trains on every cleaned spectrum outside the shared validation set:
#   stage_1_5pct    5% of each part file, split 80/20 by spectrum (Small_data_split)
#   stage_2_9pct    + 5% of the rest, same validation (Additional_data_split)
#   stage_3_100pct  + all of the rest, same validation (Additional_data_split with both fractions 1.0)
# The 100% stage holds several pandas copies of the whole set at once (about 6-8 GB each), hence the memory.
source "$SLURM_SUBMIT_DIR/config.sh"
source "$BP_DIR/common.sh"
log_hardware
match_threads_to_allocation
use_env casmi
mkdir -p "$SPLITS_DIR"

run_nb "$REPO/MichaelM/Small_data_split.ipynb" Small_data_split-bluepebble \
    "folder_name=\"$COMPETITION-derived\"" \
    "OUTPUT_DIR=Path(\"$SPLITS_DIR/stage_1_5pct\")"

for stage in 9pct 100pct; do
    if [[ $stage == 9pct ]]; then extra=(); out=stage_2_9pct
    else extra=(ADDITIONAL_SAMPLE_FRACTION=1.0 TRAIN_FRACTION=1.0); out=stage_3_100pct; fi
    run_nb "$REPO/MichaelM/Additional_data_split.ipynb" "Additional_data_split-bluepebble-$stage" \
        "SPECTRA_DIR=Path(\"$DERIVED_DIR/spectra_train\")" \
        "PREVIOUS_DATA_DIR=Path(\"$SPLITS_DIR/stage_1_5pct\")" \
        "OUTPUT_DIR=Path(\"$SPLITS_DIR/$out\")" \
        ${extra[@]+"${extra[@]}"}
done

python - <<EOF
import pyarrow.parquet as pq
# Michael's numbers for the first two stages (his experiment metrics); the third is new.
expected = {"stage_1_5pct": (91851, 22963), "stage_2_9pct": (179108, 22963), "stage_3_100pct": (None, 22963)}
for stage, (train, val) in expected.items():
    n_train = pq.ParquetFile("$SPLITS_DIR/" + stage + "/train.parquet").metadata.num_rows
    n_val = pq.ParquetFile("$SPLITS_DIR/" + stage + "/validation.parquet").metadata.num_rows
    note = "" if train in (None, n_train) and val == n_val else "   !! differs from Michael's split"
    print(f"{stage:>15}: {n_train:>9,} training, {n_val:>6,} validation spectra{note}")
EOF
