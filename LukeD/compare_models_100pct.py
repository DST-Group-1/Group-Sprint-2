#!/usr/bin/env python3
"""Compare the three Sprint 2 models on the same validation spectra (100% training split).

Run from the repo root (Group-Sprint-2) once the BluePebble results are in place:

    python LukeD/compare_models_100pct.py

Prints the comparison table and writes it to LukeD/results/model_comparison_100pct.csv.

Task: rank the true molecule of each validation spectrum among all 271,040 molecules in the
100% training split (model_datasets/stage_3_100pct/train.parquet, 2,273,309 spectra).

- Fine-tuned DreaMS: closed-set classifier over the training molecules. Its predictions file
  records only the top prediction, so it gets top-1 only. It drops the 183 validation spectra
  whose molecule is not in training, leaving 22,780 of 22,963.
- Fingerprint MLP: Thomas's "known molecules, all training molecules" setting, which ranks the
  same 271,040 molecules. Joined to DreaMS row by row on (row_id, InChIKey14).
- MS2DeepScore: pretrained embeddings, nearest-neighbour search over the training spectra.
  David's notebook scores it only on the 10,029 validation spectra whose molecule appears in
  the 5% training sample. Its file has no row_id, so the shared subset is matched by molecule
  and checked by the number of spectra per molecule (they must agree for every molecule).
"""

import argparse
from pathlib import Path

import pandas as pd

DREAMS = "MichaelM/results_bluepebble/dreams_100pct_3epochs/validation_predictions.csv"
MLP = "ThomasF/results_bluepebble/mlp/group_stage_3_100pct/val_spectra.parquet"
MS2DS = "DavidA/results_bluepebble/100pct/100pct_predictions.csv"
MLP_KNOWN = "known molecules, all training molecules"
MLP_UNSEEN = "unseen molecules, training + validation molecules"


def as_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def row(subset, model, df, mol_col, hits):
    """One table row: per-spectrum hit rates plus top-1 averaged per molecule."""
    out = {
        "subset": subset,
        "model": model,
        "spectra": len(df),
        "molecules": df[mol_col].nunique(),
    }
    for k in (1, 5, 25):
        out[f"top{k}"] = df[hits[k]].mean() if k in hits else float("nan")
    out["top1_molecule_averaged"] = df.groupby(mol_col)[hits[1]].mean().mean()
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path("."),
        help="repo root (default: current directory)",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("LukeD/results/model_comparison_100pct.csv")
    )
    args = ap.parse_args()
    repo = args.repo

    dreams = pd.read_csv(repo / DREAMS, dtype={"source_part": str})
    dreams["hit1"] = as_bool(dreams["correct"]).astype(float)

    mlp_all = pd.read_parquet(repo / MLP)
    mlp = mlp_all[mlp_all.setting == MLP_KNOWN]
    mlp_unseen = mlp_all[mlp_all.setting == MLP_UNSEEN]

    ms = pd.read_csv(repo / MS2DS)
    cands = ms.predicted_top25.fillna("").str.split(";")
    for k in (1, 5, 25):
        ms[f"hit{k}"] = [float(t in c[:k]) for t, c in zip(ms.target_inchikey14, cands)]
    assert (ms.hit1 == (ms.top1 == ms.target_inchikey14)).all(), (
        "MS2DS top1 column disagrees with its top-25 list"
    )

    # MLP and DreaMS on every DreaMS-eligible spectrum, matched row by row.
    j = dreams.merge(
        mlp[["row_id", "inchikey14", "hit1", "hit5", "hit25"]],
        left_on=["row_id", "true_inchikey14"],
        right_on=["row_id", "inchikey14"],
        how="inner",
        suffixes=("_dreams", "_mlp"),
        validate="one_to_one",
    )
    assert len(j) == len(dreams) == len(mlp), (
        f"row match failed: {len(j)} matched, {len(dreams)} DreaMS, {len(mlp)} MLP"
    )

    # The MS2DeepScore subset: same molecules, and the same number of spectra for each one.
    sub = j[j.true_inchikey14.isin(set(ms.target_inchikey14))]
    counts = pd.concat(
        [
            ms.target_inchikey14.value_counts().rename("ms2ds"),
            sub.true_inchikey14.value_counts().rename("shared"),
        ],
        axis=1,
    ).fillna(0)
    assert (counts.ms2ds == counts.shared).all(), (
        "MS2DeepScore was not scored on a subset of these spectra"
    )

    full = f"all DreaMS-eligible spectra ({len(j):,})"
    shared = f"MS2DeepScore subset ({len(sub):,})"
    rows = [
        row(
            full,
            "Fingerprint MLP",
            j,
            "true_inchikey14",
            {1: "hit1_mlp", 5: "hit5", 25: "hit25"},
        ),
        row(full, "Fine-tuned DreaMS", j, "true_inchikey14", {1: "hit1_dreams"}),
        row(
            shared,
            "MS2DeepScore",
            ms,
            "target_inchikey14",
            {1: "hit1", 5: "hit5", 25: "hit25"},
        ),
        row(
            shared,
            "Fingerprint MLP",
            sub,
            "true_inchikey14",
            {1: "hit1_mlp", 5: "hit5", 25: "hit25"},
        ),
        row(shared, "Fine-tuned DreaMS", sub, "true_inchikey14", {1: "hit1_dreams"}),
    ]
    if len(mlp_unseen):
        rows.append(
            row(
                f"unseen molecules ({len(mlp_unseen):,}; MLP only)",
                "Fingerprint MLP",
                mlp_unseen,
                "inchikey14",
                {1: "hit1", 5: "hit5", 25: "hit25"},
            )
        )

    table = pd.DataFrame(rows)
    out = args.out if args.out.is_absolute() else repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    with pd.option_context(
        "display.width", 140, "display.float_format", "{:.4f}".format
    ):
        print(table.to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
