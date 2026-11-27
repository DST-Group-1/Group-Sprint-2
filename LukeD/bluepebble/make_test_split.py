#!/usr/bin/env python3
"""Build the held-out test set for scoring the trained models, and check that it cannot have leaked.

The test set is the `split == "test"` half of the held-out molecules written by
LukeD/01-data-and-validation (heldout.parquet). Those query spectra were never written to
spectra_train, and Michael's splits are cut from spectra_train, so no model trained on his splits
should have seen them. This script checks that against the split the models were actually trained
on, then writes the test spectra in the same column layout as that split's validation.parquet.

The test set is for scoring the finished models once, at the end. Never train on it, and never use
it to pick an epoch or a setting: that is what the validation set is for.

Writes to --out:
  test.parquet            the test query spectra, with exactly the columns of the split's
                          validation.parquet, in the same order
  test_molecules.parquet  one row per test molecule: class, pool, structure, whether it may be
                          offered as a candidate, its number of test spectra, and its number of
                          spectra in the training split. A molecule with none cannot be predicted
                          by a model that only outputs training molecules (DreaMS's classifier,
                          MS2DeepScore's library search), so it counts as a miss for them.
  manifest.json           counts, and the result of every check
  README.txt

    python make_test_split.py --derived <derived dir> --split <splits>/stage_3_100pct \
        --original <competition dir>/train.parquet --out <splits>/test_heldout
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

LABEL_COLUMNS = ["row_id", "inchikey14", "sim_class", "pool"]


def row_group_of(row_ids: np.ndarray, original: Path | None, spectra_train: Path) -> tuple[np.ndarray, str]:
    """The row group of the original train.parquet each spectrum came from: Michael's `source_part`.

    LukeD/01 numbers spectra by their row in the original file and writes one spectra_train part
    per row group, so part-007.parquet holds row group 7 and source_part "007" means row group 7.
    """
    if original is not None and original.exists():
        meta = pq.ParquetFile(original).metadata
        sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        if row_ids.max() >= sum(sizes):
            raise SystemExit(f"a row_id is past the end of {original}: is it the file LukeD/01 read?")
        return np.searchsorted(starts, row_ids, side="right") - 1, f"exact, from the row groups of {original}"
    # Without the original file: each part file's smallest training row_id marks its row group's start.
    firsts, parts = [], []
    for path in sorted(spectra_train.glob("part-*.parquet")):
        first = pl.scan_parquet(path).select(pl.col("row_id").min()).collect().item()
        if first is not None:
            firsts.append(first)
            parts.append(int(path.stem.split("-")[1]))
    position = np.searchsorted(np.array(firsts), row_ids, side="right") - 1
    if (position < 0).any():
        raise SystemExit("a test spectrum comes before the first training spectrum: pass --original")
    return np.array(parts)[position], "inferred from the spectra_train part files (pass --original for exact)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived", type=Path, required=True,
                    help="LukeD/01's output folder: heldout.parquet, molecules.parquet, spectra_train/")
    ap.add_argument("--split", type=Path, required=True,
                    help="the split the models trained on, holding train.parquet and validation.parquet")
    ap.add_argument("--original", type=Path, help="the competition's train.parquet, for exact source_part values")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    started = time.perf_counter()

    heldout = pl.read_parquet(args.derived / "heldout.parquet")
    molecules = pl.read_parquet(args.derived / "molecules.parquet")
    test = heldout.filter(pl.col("split") == "test")
    if test.height == 0:
        raise SystemExit("heldout.parquet has no rows with split == 'test'")

    # Only the columns the checks need, so the 2.27 million training spectra stay small in memory.
    # row_id is cast on both sides because a pandas round trip can change its integer type.
    needed = [pl.col("row_id").cast(pl.Int64), "inchikey14", "instrument_family"]
    train = pl.scan_parquet(args.split / "train.parquet").select(needed).collect()
    validation = pl.scan_parquet(args.split / "validation.parquet").select(needed).collect()
    seen = {"training split": train, "validation split": validation}
    test_ids = test.select(pl.col("row_id").cast(pl.Int64))

    # ---- Checks: nothing about a test spectrum may be in what the models trained or were tuned on.
    by_class = {k: test.filter(pl.col("sim_class").is_in(v)).select("inchikey14").unique()
                for k, v in {"2 or 3": [2, 3], "1": [1]}.items()}
    checks = {}
    for name, frame in seen.items():
        checks[f"test spectra also in the {name}"] = frame.join(test_ids, on="row_id", how="semi").height
        checks[f"spectra of class 2 or 3 test molecules in the {name}"] = (
            frame.join(by_class["2 or 3"], on="inchikey14", how="semi").height)
        checks[f"timsTOF spectra of class 1 test molecules in the {name}"] = (
            frame.filter(pl.col("instrument_family") == "timsTOF")
            .join(by_class["1"], on="inchikey14", how="semi").height)
    checks["test spectra that are not timsTOF, or are dimers"] = test.filter(
        (pl.col("instrument_family") != "timsTOF") | pl.col("is_dimer")).height
    checks["test spectra with no class"] = test.filter(pl.col("sim_class").is_null()).height
    checks["test molecules not marked as test in molecules.parquet"] = (
        test.select("inchikey14").unique()
        .join(molecules.filter(pl.col("split") == "test").select("inchikey14"), on="inchikey14", how="anti").height)
    checks["test molecules that are also validation molecules"] = (
        test.select("inchikey14").unique()
        .join(heldout.filter(pl.col("split") == "val").select("inchikey14").unique(), on="inchikey14", how="semi").height)
    for check, count in checks.items():
        print(f"{'ok ' if count == 0 else '!! '} {count:>7,}  {check}")
    failed = {k: v for k, v in checks.items() if v}
    if failed:
        raise SystemExit("!! the test set overlaps what the models saw, so it cannot be used: " + json.dumps(failed))

    # ---- The test spectra, in the layout of the split's own validation file.
    reference = pl.read_parquet(args.split / "validation.parquet", n_rows=1)
    parts, how = row_group_of(test["row_id"].to_numpy(), args.original, args.derived / "spectra_train")
    if "source_part" in reference.columns:
        as_text = reference.schema["source_part"] == pl.String     # Michael writes it as "000" to "020"
        test = test.with_columns(pl.Series("source_part", [f"{p:03d}" for p in parts] if as_text else parts))
    missing = [c for c in reference.columns if c not in test.columns]
    if missing:
        raise SystemExit(f"heldout.parquet lacks columns the split's validation.parquet has: {missing}")
    spectra = test.select(reference.columns).cast(dict(reference.schema))

    # ---- One row per test molecule, with what a closed-set model could possibly get right.
    in_training = train.group_by("inchikey14").agg(pl.len().alias("n_train_spectra_in_split"))
    test_molecules = (
        test.group_by("inchikey14").agg(pl.len().alias("n_test_spectra"))
        .join(molecules.select("inchikey14", "smiles", "molecular_formula", "mono_mass", "pool", "sim_class",
                               "in_candidates"), on="inchikey14", how="left")
        .join(in_training, on="inchikey14", how="left")
        .with_columns(pl.col("n_train_spectra_in_split").fill_null(0))
        .with_columns((pl.col("n_train_spectra_in_split") > 0).alias("in_training_split"))
        .sort("sim_class", "inchikey14")
    )
    summary = (
        test_molecules.group_by("sim_class")
        .agg(pl.len().alias("molecules"), pl.col("n_test_spectra").sum().alias("spectra"),
             pl.col("in_training_split").sum().alias("molecules in the training split"),
             (pl.col("pool") == "natural-product examples").sum().alias("natural-product molecules"))
        .sort("sim_class")
    )

    args.out.mkdir(parents=True, exist_ok=True)
    spectra.write_parquet(args.out / "test.parquet")
    test_molecules.write_parquet(args.out / "test_molecules.parquet")
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"heldout": str(args.derived / "heldout.parquet"), "split": str(args.split)},
        "source_part": how,
        "test_spectra": spectra.height,
        "test_molecules": test_molecules.height,
        "by_class": summary.to_dicts(),
        "checks": checks,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (args.out / "README.txt").write_text(
        "The held-out TEST set (LukeD/01's heldout.parquet, split == 'test'), built by make_test_split.py.\n"
        "Score finished models on it once. Never train on it or use it to choose an epoch or a setting.\n"
        "test.parquet has the columns of the training split's validation.parquet. Scores are reported\n"
        "per test molecule and per class (sim_class in test_molecules.parquet).\n")

    print(f"\n{spectra.height:,} test spectra of {test_molecules.height:,} molecules -> {args.out}")
    print(f"source_part: {how}")
    print(summary)
    print(f"done in {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
