#!/usr/bin/env python3
"""Score MS2DeepScore on the held-out test set once, with the reference library of the 100% run.

MS2DeepScore is not trained here or anywhere in the project: the pretrained model embeds spectra and
each query is answered by the molecules of its most similar reference spectra. This script reuses
David's notebook (DavidA/MS2DeepScore, as run by job 05) line for line where it matters: the same
make_spectrum, the same model loading, the same batch sizes, and the same search (cosine similarity,
the 512 nearest reference spectra, their first 25 distinct molecules). The reference library is the
embeddings job 05 already saved (every spectrum of the 100% training split), so nothing is re-embedded.

David's notebook cannot score the test set itself: its leak check requires every evaluated spectrum to
be found in the reference files, and test spectra are, correctly, not there.

Before it looks at the test set it checks itself: it repeats the 100% run's validation search and
compares the answers with that run's saved 100pct_predictions.csv. If they differ it stops.

A test molecule with no reference spectra (all of class 2 and 3, and some of class 1) cannot be found
by a library search, so it counts as a miss, as it does for DreaMS.

Writes to --out the same files as dreams_score_test.py: test_predictions.csv, test_molecules.csv
(each candidate scored by its best similarity over the molecule's spectra), test_metrics.json and
test_metrics_by_group.csv.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from matchms import Spectrum
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model

EMBED_BATCH = 256       # as in David's notebook
QUERY_BATCH = 16
SPECTRUM_TOPK = 512
TOP_K = 25
COLUMNS = ["inchikey14", "mzs", "intensities", "precursor_mz", "ionization_mode", "charge", "source_part", "row_id"]


# ---- From David's notebook, unchanged ------------------------------------------------------------
def interpret_ionmode(mode, charge):
    s = str(mode).strip().lower()
    if s in ('positive','pos','+','esi+','positive ion mode') or ('+' in s and '-' not in s): return 'positive'
    if s in ('negative','neg','-','esi-','negative ion mode') or ('-' in s and '+' not in s): return 'negative'
    try:
        c = float(charge)
        if c > 0: return 'positive'
        if c < 0: return 'negative'
    except (TypeError, ValueError): pass
    return None


def make_spectrum(row):
    mz = np.asarray(row.mzs, dtype=np.float64).reshape(-1)
    intens = np.asarray(row.intensities, dtype=np.float64).reshape(-1)
    if len(mz) != len(intens) or len(mz)==0: return None
    mask = np.isfinite(mz) & np.isfinite(intens) & (mz > 0) & (intens > 0)
    mz, intens = mz[mask], intens[mask]
    if len(mz) == 0: return None
    order = np.argsort(mz)
    mz, intens = mz[order], intens[order]
    intens = intens / intens.max()
    precursor = float(row.precursor_mz) if pd.notna(row.precursor_mz) else None
    ionmode = interpret_ionmode(row.ionization_mode, getattr(row, 'charge', None))
    if precursor is None or not np.isfinite(precursor) or ionmode is None: return None
    return Spectrum(mz=mz, intensities=intens,
                    metadata={'precursor_mz':precursor,'ionmode':ionmode},
                    metadata_harmonization=True)
# ------------------------------------------------------------------------------------------------


def read(path, columns):
    available = set(pq.ParquetFile(str(path)).schema_arrow.names)
    return pd.read_parquet(path, columns=[c for c in columns if c in available])


def embed(ms2ds, spectra):
    """As the notebook's validation cell: EMBED_BATCH spectra at a time."""
    chunks = [np.asarray(ms2ds.get_embedding_array(spectra[i:i + EMBED_BATCH]), dtype=np.float32)
              for i in range(0, len(spectra), EMBED_BATCH)]
    out = np.concatenate(chunks)
    if out.ndim != 2 or out.shape[0] != len(spectra) or not np.isfinite(out).all():
        raise ValueError("MS2DeepScore returned invalid embeddings")
    return out


@torch.no_grad()
def search(query_emb, ref_tensor, ref_labels, device):
    """For each query: its first 25 distinct molecules among the 512 nearest reference spectra (the
    notebook's search), and every distinct molecule among those 512 with its best similarity."""
    ranked_all, best_all = [], []
    k = min(SPECTRUM_TOPK, ref_tensor.shape[0])
    for start in range(0, len(query_emb), QUERY_BATCH):
        q = F.normalize(torch.as_tensor(query_emb[start:start + QUERY_BATCH], dtype=torch.float32, device=device), dim=1)
        top = torch.topk(q @ ref_tensor.T, k=k, dim=1)
        for idx_row, score_row in zip(top.indices.cpu().numpy(), top.values.cpu().numpy()):
            best = {}
            for idx, score in zip(idx_row, score_row):        # highest similarity first
                mol = ref_labels[int(idx)].decode("utf-8")
                if mol not in best:
                    best[mol] = float(score)
            ranked_all.append(list(best)[:TOP_K])
            best_all.append(best)
    return ranked_all, best_all


def rank_of(true, ranked):
    return ranked.index(true) + 1 if true in ranked else 0


def scores(ranks):
    r = np.asarray(ranks)
    return {f"top{k}": float(((r >= 1) & (r <= k)).mean()) for k in (1, 5, 25)} | {
        "mrr25": float(np.where(r >= 1, 1.0 / np.maximum(r, 1), 0.0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="ms2deepscore_model.pt, as job 05 used it")
    ap.add_argument("--reference-dir", type=Path, required=True,
                    help="job 05's output folder holding 100pct_embeddings.npy, 100pct_labels.npy, 100pct_info.json")
    ap.add_argument("--reference-tag", default="100pct")
    ap.add_argument("--check-data-dir", type=Path, required=True,
                    help="the DATA_DIR job 05 used (stage_1_5pct): its validation spectra are re-searched")
    ap.add_argument("--check-predictions", type=Path, required=True, help="job 05's 100pct_predictions.csv")
    ap.add_argument("--test", type=Path, required=True, help="make_test_split.py's output folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-agreement", type=float, default=0.995)
    args = ap.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print("device:", gpu or "CPU (no GPU found: the search will be slow)")

    # ---- The model and the saved reference library.
    model = load_model(str(args.model))
    if torch.cuda.is_available() and hasattr(model, "to"):
        model = model.to("cuda")
    if hasattr(model, "eval"):
        model.eval()
    ms2ds = MS2DeepScore(model)

    tag = args.reference_tag
    info = json.loads((args.reference_dir / f"{tag}_info.json").read_text())
    assert info["excluded_validation"], "the reference library was built without removing the validation spectra"
    assert info["checkpoint_path"] == str(args.model.resolve()), \
        f"the reference embeddings were made with {info['checkpoint_path']}, not {args.model.resolve()}"
    count = info["count"]
    reference = np.load(args.reference_dir / f"{tag}_embeddings.npy", mmap_mode="r")[:count]
    ref_labels = np.load(args.reference_dir / f"{tag}_labels.npy", mmap_mode="r")[:count]
    print(f"reference library: {count:,} spectra from {info['source_files']}, "
          f"{info['excluded_count']:,} validation spectra excluded")
    ref_tensor = F.normalize(torch.from_numpy(np.asarray(reference)).to(device), dim=1)
    reference_molecules = {x.decode("utf-8") for x in np.unique(ref_labels)}

    # ---- Self-check: job 05's validation search, repeated, must give job 05's answers.
    train5 = read(args.check_data_dir / "train.parquet", ["inchikey14"])
    validation_all = read(args.check_data_dir / "validation.parquet", COLUMNS)
    known = set(train5["inchikey14"].dropna().astype(str))
    valid = validation_all.loc[validation_all["inchikey14"].astype(str).isin(known)]
    sp, val_true = [], []
    for r in valid.itertuples(index=False):
        s = make_spectrum(r)
        if s is None:
            continue
        sp.append(s)
        val_true.append(str(r.inchikey14))
    val_ranked, _ = search(embed(ms2ds, sp), ref_tensor, ref_labels, device)
    saved = pd.read_csv(args.check_predictions, keep_default_na=False)
    same_rows = len(saved) == len(val_true) and (saved["target_inchikey14"].astype(str).to_numpy() == np.asarray(val_true)).all()
    top1_now = [p[0] if p else "" for p in val_ranked]
    agreement = float(np.mean(saved["top1"].astype(str).to_numpy() == np.asarray(top1_now))) if same_rows else 0.0
    list_agreement = float(np.mean(saved["predicted_top25"].astype(str).to_numpy()
                                   == np.asarray([";".join(p) for p in val_ranked]))) if same_rows else 0.0
    self_check = {"validation_spectra": len(val_true), "saved_spectra": len(saved), "same_spectra_in_same_order": bool(same_rows),
                  "top1_agreement": agreement, "top25_list_agreement": list_agreement,
                  "top1_now": float(np.mean([t == p for t, p in zip(val_true, top1_now)])),
                  "top1_saved": float((saved["top1"].astype(str) == saved["target_inchikey14"].astype(str)).mean())}
    print(f"self-check: {len(val_true):,} validation spectra (saved: {len(saved):,}), same order: {same_rows}; "
          f"top-1 agrees on {agreement:.2%}, whole top-25 list on {list_agreement:.2%}; "
          f"top-1 {self_check['top1_now']:.4f} now, {self_check['top1_saved']:.4f} saved")
    if not same_rows or agreement < args.min_agreement:
        raise SystemExit("!! this does not reproduce job 05's validation results, so the test set was NOT scored: "
                         + json.dumps(self_check))

    # ---- The test set, searched once.
    test = read(args.test / "test.parquet", COLUMNS)
    labels = pd.read_parquet(args.test / "test_molecules.parquet", columns=["inchikey14", "sim_class", "pool"])
    test["inchikey14"] = test["inchikey14"].astype(str)
    test = test.merge(labels, on="inchikey14", how="left", validate="many_to_one")
    assert test["sim_class"].notna().all(), "a test spectrum has no class"
    test = test.sort_values(["inchikey14", "row_id"]).reset_index(drop=True)
    test["in_training"] = test["inchikey14"].isin(reference_molecules)

    spectra, keep = [], []
    for i, r in enumerate(test.itertuples(index=False)):
        s = make_spectrum(r)
        if s is not None:
            spectra.append(s)
            keep.append(i)
    test["embedded"] = False
    test.loc[keep, "embedded"] = True
    ranked, best = search(embed(ms2ds, spectra), ref_tensor, ref_labels, device) if spectra else ([], [])
    test["predicted_top25"], test["n_unique_candidates"] = "", 0
    test.loc[keep, "predicted_top25"] = [";".join(p) for p in ranked]
    test.loc[keep, "n_unique_candidates"] = [len(p) for p in ranked]
    test["top1"] = test["predicted_top25"].str.split(";").str[0]
    test["rank"] = [rank_of(t, p.split(";") if p else []) for t, p in zip(test["inchikey14"], test["predicted_top25"])]
    test["correct"] = test["rank"] == 1
    print(f"test: {len(test):,} spectra, {len(spectra):,} usable by MS2DeepScore")

    # Per molecule: each candidate's best similarity over the molecule's spectra.
    best_of = dict(zip(keep, best))
    mol_rows = []
    for mol, rows in test.groupby("inchikey14", sort=False).groups.items():
        pooled = {}
        for i in rows:
            for cand, score in best_of.get(i, {}).items():
                if score > pooled.get(cand, -np.inf):
                    pooled[cand] = score
        order = sorted(pooled, key=lambda c: -pooled[c])[:TOP_K]
        first = test.loc[rows[0]]
        mol_rows.append({"true_inchikey14": mol, "sim_class": first["sim_class"], "pool": first["pool"],
                         "n_spectra": len(rows), "in_training": bool(first["in_training"]), "rank": rank_of(mol, order),
                         "predicted_inchikey14": order[0] if order else "", "predicted_top25": ";".join(order)})
    molecules = pd.DataFrame(mol_rows)

    groups = [("all", None)] + [(f"class {int(c)}", ("sim_class", c)) for c in sorted(test["sim_class"].unique())] + [
        ("natural-product examples", ("pool", "natural-product examples")), ("molecules with training spectra", ("in_training", True))]
    rows = []
    for name, key in groups:
        s = test if key is None else test[test[key[0]] == key[1]]
        m = molecules if key is None else molecules[molecules[key[0]] == key[1]]
        if len(s) == 0:
            continue
        rows.append({"group": name, "spectra": len(s), "molecules": len(m),
                     "molecules_with_training_spectra": int(m["in_training"].sum()),
                     **{f"spectrum_{k}": v for k, v in scores(s["rank"]).items()},
                     "spectrum_top1_molecule_averaged": float(s.groupby("inchikey14")["correct"].mean().mean()),
                     **{f"molecule_{k}": v for k, v in scores(m["rank"]).items()}})
    table = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    test.rename(columns={"inchikey14": "true_inchikey14"})[
        ["row_id", "source_part", "true_inchikey14", "sim_class", "pool", "in_training", "embedded", "top1", "correct",
         "rank", "n_unique_candidates", "predicted_top25"]].to_csv(args.out / "test_predictions.csv", index=False)
    molecules.to_csv(args.out / "test_molecules.csv", index=False)
    table.to_csv(args.out / "test_metrics_by_group.csv", index=False)
    metrics = {
        "experiment": "Pretrained MS2DeepScore, 100% training split as the reference library, on the held-out test set, scored once",
        "model": str(args.model), "reference": {k: info[k] for k in ("count", "source_files", "excluded_count")},
        "test_set": str(args.test), "model_retrained": False,
        "notes": {"closed_set": "a molecule with no reference spectra cannot be found; such test molecules are misses",
                  "rank": "rank of the true molecule among the first 25 distinct molecules, 0 if not there",
                  "molecule_scores": "each candidate scored by its best similarity over the molecule's spectra"},
        "self_check": self_check, "by_group": table.to_dict(orient="records"),
        "seconds": time.time() - started, "gpu": gpu, "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
    }
    (args.out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table[["group", "spectra", "molecules", "molecules_with_training_spectra", "spectrum_top1",
                     "spectrum_top25", "molecule_top1", "molecule_mrr25"]])
    print(f"written to {args.out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
