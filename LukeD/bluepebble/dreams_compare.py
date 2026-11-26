"""Finish the DreaMS 100% run's comparison with the earlier runs, without retraining.

Michael's notebook compares runs by joining their validation predictions on row_id and then
asserting that the class *indices* agree. Each split numbers its classes in its own
class_mapping.json, so that assertion only holds when the later split extended the earlier
split's mapping. Here the labels are compared as InChIKey-14 strings and rows are matched on
(source_part, row_id), which does not depend on any mapping.

It also checks whether BluePebble's regenerated 5% split is the same as Michael's laptop one,
because the 100% split's validation set is that 5% split's validation set.

    python dreams_compare.py --runs 5pct=<dir> 9pct=<dir> 100pct=<dir> \
        --split-5pct <BluePebble stage_1_5pct> --mapping-5pct <Michael's 5% class_mapping.json> \
        --out <dir for the comparison files>

Each run dir holds that run's validation_predictions.csv and metrics.json. The first run is the
reference: the shared rows are its validation spectra that every other run also scored.
"""
import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# Michael's runs print this too: some predicted molecules are not among the true ones.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def calculate_metrics(labels, predictions):
    # The same definitions as calculate_metrics in Michael's notebook.
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
        "macro_f1_true_classes": f1_score(
            labels, predictions, labels=np.unique(labels), average="macro", zero_division=0
        ),
    }


def part_number(value):
    digits = re.findall(r"\d+", str(value))
    if not digits:
        raise ValueError(f"source_part {value!r} has no number in it")
    return int(digits[-1])


def load_predictions(run_dir):
    df = pd.read_csv(run_dir / "validation_predictions.csv", dtype={"source_part": str, "row_id": str})
    missing = {"source_part", "row_id", "true_inchikey14", "predicted_inchikey14"} - set(df.columns)
    if missing:
        raise SystemExit(f"{run_dir}/validation_predictions.csv has no {sorted(missing)} column")
    df["key"] = list(zip(df["source_part"].map(part_number), df["row_id"].astype(np.int64)))
    if df["key"].duplicated().any():
        raise SystemExit(f"{run_dir}: (source_part, row_id) is not unique")
    return df.set_index("key")


def check_5pct_split(split_dir, mapping_path, reference):
    print("== Is BluePebble's 5% split the same as Michael's laptop one?")
    if mapping_path and mapping_path.exists() and (split_dir / "class_mapping.json").exists():
        laptop = json.loads(mapping_path.read_text())
        bluepebble = json.loads((split_dir / "class_mapping.json").read_text())
        same_classes = set(laptop) == set(bluepebble)
        same_indices = same_classes and all(laptop[k] == bluepebble[k] for k in laptop)
        print(f"   classes: laptop {len(laptop):,}, BluePebble {len(bluepebble):,}; "
              f"same classes: {same_classes}; same indices: {same_indices}")
    else:
        print("   (class mappings not both found; skipped)")
    validation_path = split_dir / "validation.parquet"
    if validation_path.exists():
        validation = pd.read_parquet(validation_path, columns=["source_part", "row_id"])
        keys = set(zip(validation["source_part"].map(part_number), validation["row_id"].astype(np.int64)))
        inside = sum(k in keys for k in reference.index)
        print(f"   the reference run's {len(reference):,} validation spectra: {inside:,} are in "
              f"BluePebble's 5% validation set ({len(keys):,} spectra before filtering)")
    else:
        print(f"   ({validation_path} not found; skipped)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--split-5pct", type=Path)
    parser.add_argument("--mapping-5pct", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for item in args.runs:
        name, _, path = item.partition("=")
        runs[name] = Path(path)
    names = list(runs)
    predictions = {name: load_predictions(path) for name, path in runs.items()}
    reference = predictions[names[0]]
    for name in names:
        print(f"{name}: {len(predictions[name]):,} validation spectra scored ({runs[name]})")

    if args.split_5pct:
        check_5pct_split(args.split_5pct, args.mapping_5pct, reference)

    # What the notebook did: join on row_id alone and compare the class indices.
    last = predictions[names[-1]]
    by_row = reference.reset_index().merge(last.reset_index(), on="row_id", suffixes=("_ref", "_last"))
    print(f"== The notebook's check ({names[0]} vs {names[-1]}, joined on row_id): {len(by_row):,} rows, "
          f"{(by_row['true_label_index_ref'] != by_row['true_label_index_last']).sum():,} with different "
          f"class indices, {(by_row['true_inchikey14_ref'] != by_row['true_inchikey14_last']).sum():,} with "
          f"different InChIKeys, {(by_row['source_part_ref'].map(part_number) != by_row['source_part_last'].map(part_number)).sum():,} "
          f"from different source parts")

    # The comparison done here: the same spectrum, matched on (source_part, row_id), by InChIKey.
    shared = reference.index
    for name in names[1:]:
        shared = shared.intersection(predictions[name].index)
    truth = pd.DataFrame({name: predictions[name].loc[shared, "true_inchikey14"] for name in names})
    agree = truth.nunique(axis=1) == 1
    print(f"== Shared spectra (scored by every run, matched on source_part and row_id): {len(shared):,} "
          f"of the reference run's {len(reference):,}; true molecule the same in every run: {int(agree.sum()):,}")
    if not agree.all():
        print(f"!! {int((~agree).sum()):,} shared spectra have different true labels between runs; left out")
        shared = shared[agree.to_numpy()]
    if len(shared) == 0:
        raise SystemExit("no shared spectra to compare")

    labels = reference.loc[shared, "true_inchikey14"].to_numpy()
    rows = []
    for name in names:
        metrics = calculate_metrics(labels, predictions[name].loc[shared, "predicted_inchikey14"].to_numpy())
        rows.append({"experiment": name, "shared_spectra": len(shared), **metrics})
    shared_comparison = pd.DataFrame(rows)

    scaling_rows = []
    for name, path in runs.items():
        m = json.loads((path / "metrics.json").read_text()) if (path / "metrics.json").exists() else {}
        scaling_rows.append({
            "experiment": name,
            "training_spectra": m.get("training_spectra"),
            "training_classes": m.get("number_of_training_classes"),
            "validation_spectra": m.get("validation_spectra"),
            "accuracy_full_eligible_validation": m.get("accuracy"),
            "balanced_accuracy_full_eligible_validation": m.get("balanced_accuracy"),
            "macro_f1_full_eligible_validation": m.get("macro_f1"),
            "weighted_f1_full_eligible_validation": m.get("weighted_f1"),
            "best_epoch": m.get("best_epoch"),
            "training_time_hours": m.get("total_training_time_hours"),
            "average_epoch_time_seconds": m.get("average_epoch_time_seconds"),
            "peak_gpu_memory_allocated_gb": m.get("peak_gpu_memory_allocated_gb"),
            "gpu": m.get("gpu"),
        })
    scaling = pd.DataFrame(scaling_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    shared_comparison.to_csv(args.out / "shared_validation_comparison.csv", index=False)
    scaling.to_csv(args.out / "experiment_scaling_comparison.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n== Same spectra for every run (shared_validation_comparison.csv)")
        print(shared_comparison.round(4).to_string(index=False))
        print("\n== Each run on its own validation set (experiment_scaling_comparison.csv)")
        print(scaling.round(4).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = shared_comparison.set_index("experiment")[
            ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
        ].plot(kind="bar", figsize=(11, 6))
        ax.set_title(f"Fine-tuned DreaMS on the same {len(shared):,} validation spectra")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.grid(axis="y")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(args.out / "shared_validation_metric_comparison.png", dpi=300, bbox_inches="tight")
    except ImportError:
        pass
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
