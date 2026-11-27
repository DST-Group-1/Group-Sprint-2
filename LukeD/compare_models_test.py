#!/usr/bin/env python3
"""Compare the three Sprint 2 models on the held-out test set (jobs 07 to 10 on BluePebble).

Run from the repo root (Group-Sprint-2) once the test results are committed:

    python LukeD/compare_models_test.py

Every model was trained on the 100% group split and scored once on the same test spectra: the test
half of LukeD/01's held-out molecules (heldout.parquet, split == "test"), 2,998 molecules in three
classes. Unlike the shared validation set, which is split by spectrum, these are split by molecule:

    class 1  the molecule's timsTOF spectra are held out; its library spectra may be in training
    class 2  every spectrum is held out; its structure can still be offered as a candidate
    class 3  every spectrum is held out and its structure is withheld from the candidates

DreaMS (a classifier over the training molecules) and MS2DeepScore (a search of the training
spectra) can only answer with a molecule that has training spectra, so class 2 and 3 molecules are
misses for them by construction. The like-for-like comparison is therefore "training molecules" as
the candidates for all three models. The fingerprint MLP can also rank molecules it never saw, so it
has extra rows with the test molecules added to the candidates (class 3 withheld), and the final
model of Thomas's own protocol (per molecule, ±10 ppm candidates) has a table of its own.

Writes LukeD/results/model_comparison_test.csv (every model, candidate setting and group) and
LukeD/results/model_comparison_test_final_mlp.csv.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

DREAMS = Path("MichaelM/results_bluepebble/dreams_100pct_3epochs_test")
MS2DS = Path("DavidA/results_bluepebble/100pct_test")
MLP = Path("ThomasF/results_bluepebble/mlp_test")
CLOSED = "training molecules"
GROUPS = ["all", "class 1", "class 2", "class 3", "natural-product examples", "molecules with training spectra"]
METRICS = ["spectrum_top1", "spectrum_top5", "spectrum_top25", "spectrum_mrr25", "spectrum_top1_molecule_averaged",
           "molecule_top1", "molecule_top5", "molecule_top25", "molecule_mrr25"]


def load(repo):
    tables, spectra, checks = [], {}, {}
    for model, folder in [("Fine-tuned DreaMS", DREAMS), ("MS2DeepScore", MS2DS)]:
        if not (repo / folder / "test_metrics_by_group.csv").exists():
            print(f"!! no test results for {model} in {folder}: skipped")
            continue
        t = pd.read_csv(repo / folder / "test_metrics_by_group.csv").assign(model=model, candidates=CLOSED)
        tables.append(t)
        spectra[model] = set(pd.read_csv(repo / folder / "test_predictions.csv", usecols=["row_id"])["row_id"].astype(int))
        checks[model] = json.loads((repo / folder / "test_metrics.json").read_text())["self_check"]
    final = None
    if (repo / MLP / "group_test_metrics_by_group.csv").exists():
        t = pd.read_csv(repo / MLP / "group_test_metrics_by_group.csv").rename(columns={"setting": "candidates"})
        tables.append(t.drop(columns=["scoring", "answer_among_candidates"]).assign(model="Fingerprint MLP"))
        per_spectrum = pd.read_parquet(repo / MLP / "group_test_spectra.parquet", columns=["row_id", "setting"])
        spectra["Fingerprint MLP"] = set(per_spectrum.loc[per_spectrum["setting"] == CLOSED, "row_id"].astype(int))
        summary = json.loads((repo / MLP / "test_summary.json").read_text())
        checks["Fingerprint MLP (group split)"] = summary["group_model"]["self_check"]
        checks["Fingerprint MLP (final)"] = summary["final_model"]["self_check_val_mrr"]
        final = summary["final_model"]["test"]
    else:
        print(f"!! no test results for the fingerprint MLP in {MLP}: skipped")
    if not tables:
        raise SystemExit("no test results found: run from the repo root after jobs 08 to 10")
    return pd.concat(tables, ignore_index=True), spectra, checks, final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, default=Path("."), help="repo root (default: current directory)")
    ap.add_argument("--out", type=Path, default=Path("LukeD/results/model_comparison_test.csv"))
    args = ap.parse_args()
    table, spectra, checks, final = load(args.repo)

    # Every model must have been scored on exactly the same test spectra.
    sets = list(spectra.values())
    assert all(s == sets[0] for s in sets), {k: len(v) for k, v in spectra.items()}
    print(f"all models scored the same {len(sets[0]):,} test spectra")
    print("self-checks against each model's own validation run:")
    for model, check in checks.items():
        print(f"  {model}: {check}")

    table = table[["model", "candidates", "group", "spectra", "molecules", "molecules_with_training_spectra", *METRICS]]
    table["group"] = pd.Categorical(table["group"], GROUPS, ordered=True)
    table = table.sort_values(["candidates", "group", "model"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    show = ["model", "group", "spectra", "molecules_with_training_spectra", "spectrum_top1", "spectrum_top5",
            "spectrum_top25", "spectrum_mrr25", "molecule_top1", "molecule_mrr25"]
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 3):
        print("\nLike for like: candidates are the training split's molecules, for every model")
        print(table[table["candidates"] == CLOSED][show].to_string(index=False))
        mlp_open = table[(table["model"] == "Fingerprint MLP") & (table["candidates"] != CLOSED)]
        if len(mlp_open):
            print("\nFingerprint MLP with other candidate sets (DreaMS and MS2DeepScore cannot use these)")
            print(mlp_open[["candidates", *show[1:]]].to_string(index=False))
        if final is not None:
            rows = [{"group": g, **{k: v for k, v in final[g].items()}} for g in final]
            final_table = pd.DataFrame(rows)
            final_table.to_csv(args.out.with_name(args.out.stem + "_final_mlp.csv"), index=False)
            print("\nFingerprint MLP, final model under its own protocol (per molecule, ±10 ppm candidates, class 3 withheld)")
            print(final_table.to_string(index=False))
    print(f"\nwritten: {args.out}" + (f", {args.out.with_name(args.out.stem + '_final_mlp.csv')}" if final else ""))


if __name__ == "__main__":
    main()
