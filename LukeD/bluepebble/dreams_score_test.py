#!/usr/bin/env python3
"""Score the fine-tuned DreaMS from the 100% run on the held-out test set, without training it again.

It rebuilds the model exactly as MichaelM/DreaMS_tuned-bluepebble-100pct.ipynb does (the pretrained
DreaMS encoder, the precursor token's 1,024 numbers, dropout, one linear layer over the training
molecules) and loads the weights his run saved as its best epoch. Spectra are prepared with the same
SpectrumPreprocessor settings, read from the checkpoint.

Before it looks at the test set it checks itself: it predicts the validation spectra again and
compares them with the predictions the 100% run saved. If they disagree it stops, because the model
or the preprocessing has not been rebuilt faithfully and test scores would be wrong.

The test set is scored once. DreaMS can only answer with a molecule it was trained on, so a test
molecule with no training spectra (all of class 2 and 3, and some of class 1) counts as a miss.

Writes to --out:
  test_predictions.csv   one row per test spectrum: true and predicted molecule, confidence, the rank
                         of the true molecule in the top 25 (empty if not there) and the top 25 itself
  test_molecules.csv     one row per test molecule: its spectra pooled by taking each candidate's
                         highest probability over them, then ranked (the competition scores molecules)
  test_metrics.json      every metric, by class and for the natural-product examples, the self-check
                         result, and run details
  test_metrics_by_group.csv  the same metrics as a table
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA

TOP_K = 25


class FineTunedDreaMSClassifier(nn.Module):
    """Copied from Michael's notebook. The attribute names must match for the saved weights to load."""

    def __init__(self, encoder, num_classes, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        embeddings = self.encoder(x)
        embeddings = embeddings[:, 0, :]
        embeddings = self.dropout(embeddings)
        return self.classifier(embeddings)


def load_checkpoint(path, device):
    # As in the notebook's "Restoring the best model" cell.
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def preprocess(df, spec_preproc, n_highest_peaks, name):
    """The notebook's preprocess_dataframe: peaks and precursor m/z into DreaMS's (61, 2) input."""
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        mzs = np.asarray(row.mzs, dtype=np.float32)
        intensities = np.asarray(row.intensities, dtype=np.float32)
        if len(mzs) != len(intensities):
            raise ValueError(f"Mismatched peak arrays in {name}, row {i}")
        if len(mzs) == 0:
            raise ValueError(f"Empty spectrum in {name}, row {i}")
        processed = np.asarray(
            spec_preproc(np.vstack([mzs, intensities]), float(row.precursor_mz), high_form=False, augment=False),
            dtype=np.float32)
        if processed.shape != (n_highest_peaks + 1, 2):
            raise ValueError(f"Unexpected preprocessed shape at {name}, row {i}: {processed.shape}")
        if not np.isfinite(processed).all():
            raise ValueError(f"Non-finite values in {name}, row {i}")
        out.append(processed)
    print(f"{name}: {len(out):,} spectra preprocessed")
    return np.stack(out)


@torch.no_grad()
def spectrum_topk(model, X, device, batch_size, k):
    """Top-k class indices and probabilities for each spectrum."""
    idx, prob = [], []
    for start in range(0, len(X), batch_size):
        logits = model(torch.from_numpy(X[start:start + batch_size]).to(device))
        p, i = torch.topk(torch.softmax(logits, dim=1), k, dim=1)
        idx.append(i.cpu().numpy())
        prob.append(p.cpu().numpy())
    return np.concatenate(idx), np.concatenate(prob)


@torch.no_grad()
def molecule_topk(model, X, molecule_codes, device, max_spectra, k):
    """Top-k classes per molecule, scoring each class by its highest probability over the molecule's spectra.

    X must be sorted so that each molecule's spectra are contiguous; molecule_codes gives each row's molecule.
    """
    bounds = np.flatnonzero(np.diff(molecule_codes)) + 1
    starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(X)]])
    idx, prob, m = [], [], 0
    while m < len(starts):
        last = m
        while last + 1 < len(starts) and ends[last + 1] - starts[m] <= max_spectra:
            last += 1
        probs = torch.softmax(model(torch.from_numpy(X[starts[m]:ends[last]]).to(device)), dim=1)
        for j in range(m, last + 1):
            pooled = probs[starts[j] - starts[m]:ends[j] - starts[m]].max(dim=0).values
            p, i = torch.topk(pooled, k)
            idx.append(i.cpu().numpy())
            prob.append(p.cpu().numpy())
        m = last + 1
    return np.stack(idx), np.stack(prob)


def rank_of(true_idx, top_idx):
    """1-based rank of each true class in its top-k row, or 0 if absent (or if the true class is -1)."""
    hit = (top_idx == true_idx[:, None]) & (true_idx[:, None] >= 0)
    return np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)


def scores(ranks):
    r = np.asarray(ranks)
    return {f"top{k}": float(((r >= 1) & (r <= k)).mean()) for k in (1, 5, 25)} | {
        "mrr25": float(np.where(r >= 1, 1.0 / np.maximum(r, 1), 0.0).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="the 100%% run's best model, dreams_100pct.pt")
    ap.add_argument("--ssl-checkpoint", type=Path, required=True, help="pretrained DreaMS, ssl_model.ckpt")
    ap.add_argument("--split", type=Path, required=True, help="the split it trained on (train.parquet, validation.parquet)")
    ap.add_argument("--test", type=Path, required=True, help="make_test_split.py's output folder")
    ap.add_argument("--reference-predictions", type=Path, required=True,
                    help="the 100%% run's validation_predictions.csv, for the self-check")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32, help="as in the notebook")
    ap.add_argument("--min-agreement", type=float, default=0.995,
                    help="share of validation predictions that must match the saved ones")
    args = ap.parse_args()
    started = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    print("device:", gpu or "CPU (no GPU found: this will be slow)")

    # ---- The model, as the notebook builds it, with the 100% run's best weights.
    checkpoint = load_checkpoint(args.checkpoint, device)
    class_to_idx = {str(k): int(v) for k, v in checkpoint["class_to_idx"].items()}
    config = checkpoint["config"]
    num_classes = len(class_to_idx)
    assert num_classes == config["num_classes"], "the checkpoint's class list and config disagree"
    k = min(TOP_K, num_classes)
    idx_to_class = np.empty(num_classes, dtype=object)
    for name, i in class_to_idx.items():
        idx_to_class[i] = name
    print(f"checkpoint: epoch {checkpoint['epoch']}, validation accuracy {checkpoint['validation_accuracy']:.4f}, "
          f"{num_classes:,} classes")

    encoder = PreTrainedModel.from_ckpt(ckpt_path=str(args.ssl_checkpoint), ckpt_cls=DreaMSModel,
                                        n_highest_peaks=config["n_highest_peaks"]).model
    model = FineTunedDreaMSClassifier(encoder, num_classes, dropout=config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])          # strict: every weight must match
    model.eval()
    del checkpoint

    spec_preproc = SpectrumPreprocessor(
        dformat=DataFormatA(), prec_intens=config["precursor_intensity"],
        n_highest_peaks=config["n_highest_peaks"], normalize_mzs=False, precision=32,
        to_relative_intensities=True)
    columns = ["row_id", "inchikey14", "precursor_mz", "mzs", "intensities"]

    # ---- Self-check on validation: the same spectra as the 100% run scored, the same predictions.
    training_molecules = set(pd.read_parquet(args.split / "train.parquet", columns=["inchikey14"])["inchikey14"].astype(str))
    validation = pd.read_parquet(args.split / "validation.parquet", columns=columns)
    validation["inchikey14"] = validation["inchikey14"].astype(str)
    validation = validation[validation["inchikey14"].isin(training_molecules)].reset_index(drop=True)
    X_val = preprocess(validation, spec_preproc, config["n_highest_peaks"], "validation")
    top_val, _ = spectrum_topk(model, X_val, device, args.batch_size, 1)
    validation["predicted_inchikey14"] = idx_to_class[top_val[:, 0]]

    reference = pd.read_csv(args.reference_predictions)
    joined = validation.merge(reference[["row_id", "true_inchikey14", "predicted_inchikey14"]], on="row_id",
                              how="outer", suffixes=("", "_saved"), validate="one_to_one", indicator=True)
    unmatched = int((joined["_merge"] != "both").sum())
    both = joined[joined["_merge"] == "both"]
    assert (both["inchikey14"] == both["true_inchikey14"]).all(), "validation labels differ from the saved run's"
    agreement = float((both["predicted_inchikey14"] == both["predicted_inchikey14_saved"]).mean())
    accuracy = float((validation["predicted_inchikey14"] == validation["inchikey14"]).mean())
    saved_accuracy = float((reference["true_inchikey14"] == reference["predicted_inchikey14"]).mean())
    self_check = {"validation_spectra": len(validation), "saved_spectra": len(reference),
                  "rows_not_in_both": unmatched, "prediction_agreement": agreement,
                  "accuracy_now": accuracy, "accuracy_saved": saved_accuracy}
    print(f"self-check: {len(both):,} validation spectra, {unmatched} unmatched, "
          f"{agreement:.2%} of predictions agree with the saved run; accuracy {accuracy:.4f} now, {saved_accuracy:.4f} saved")
    if unmatched or agreement < args.min_agreement:
        raise SystemExit("!! the rebuilt model does not reproduce the 100% run's validation predictions, "
                         "so the test set was NOT scored: " + json.dumps(self_check))
    del X_val

    # ---- The test set, scored once.
    test = pd.read_parquet(args.test / "test.parquet", columns=[*columns, "source_part"])
    labels = pd.read_parquet(args.test / "test_molecules.parquet", columns=["inchikey14", "sim_class", "pool"])
    test["inchikey14"] = test["inchikey14"].astype(str)
    test = test.merge(labels, on="inchikey14", how="left", validate="many_to_one")
    assert test["sim_class"].notna().all(), "a test spectrum has no class"
    test = test.sort_values(["inchikey14", "row_id"]).reset_index(drop=True)   # each molecule's spectra together
    test["true_label_index"] = test["inchikey14"].map(class_to_idx).fillna(-1).astype(np.int64)
    test["in_training"] = test["true_label_index"] >= 0
    X_test = preprocess(test, spec_preproc, config["n_highest_peaks"], "test")

    top_idx, top_prob = spectrum_topk(model, X_test, device, args.batch_size, k)
    test["predicted_label_index"] = top_idx[:, 0]
    test["predicted_inchikey14"] = idx_to_class[top_idx[:, 0]]
    test["prediction_confidence"] = top_prob[:, 0]
    test["rank"] = rank_of(test["true_label_index"].to_numpy(), top_idx)
    test["correct"] = test["rank"] == 1
    test["predicted_top25"] = [";".join(idx_to_class[row]) for row in top_idx]

    codes = pd.factorize(test["inchikey14"])[0]
    mol_idx, mol_prob = molecule_topk(model, X_test, codes, device, max_spectra=256, k=k)
    molecules = (test.groupby("inchikey14", sort=False)
                 .agg(sim_class=("sim_class", "first"), pool=("pool", "first"), n_spectra=("row_id", "size"),
                      true_label_index=("true_label_index", "first"), in_training=("in_training", "first"))
                 .reset_index())
    molecules["rank"] = rank_of(molecules["true_label_index"].to_numpy(), mol_idx)
    molecules["predicted_inchikey14"] = idx_to_class[mol_idx[:, 0]]
    molecules["prediction_confidence"] = mol_prob[:, 0]
    molecules["predicted_top25"] = [";".join(idx_to_class[row]) for row in mol_idx]

    # ---- Metrics: by spectrum, averaged per molecule, and per molecule with pooled spectra.
    groups = {"all": slice(None)}
    for c in sorted(test["sim_class"].unique()):
        groups[f"class {int(c)}"] = int(c)
    groups["natural-product examples"] = "natural-product examples"
    groups["molecules with training spectra"] = "in_training"
    rows = []
    for name, key in groups.items():
        if key == slice(None):
            s, m = test, molecules
        elif key == "in_training":
            s, m = test[test["in_training"]], molecules[molecules["in_training"]]
        elif isinstance(key, str):
            s, m = test[test["pool"] == key], molecules[molecules["pool"] == key]
        else:
            s, m = test[test["sim_class"] == key], molecules[molecules["sim_class"] == key]
        if len(s) == 0:
            continue
        spectrum = scores(s["rank"])
        pooled = scores(m["rank"])
        rows.append({
            "group": name, "spectra": len(s), "molecules": len(m),
            "molecules_with_training_spectra": int(m["in_training"].sum()),
            **{f"spectrum_{k}": v for k, v in spectrum.items()},
            "spectrum_top1_molecule_averaged": float(s.groupby("inchikey14")["correct"].mean().mean()),
            **{f"molecule_{k}": v for k, v in pooled.items()},
        })
    table = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    test.rename(columns={"inchikey14": "true_inchikey14"})[
        ["row_id", "source_part", "true_inchikey14", "sim_class", "pool", "in_training", "true_label_index",
         "predicted_label_index", "predicted_inchikey14", "prediction_confidence", "correct", "rank",
         "predicted_top25"]].to_csv(args.out / "test_predictions.csv", index=False)
    molecules.rename(columns={"inchikey14": "true_inchikey14"}).to_csv(args.out / "test_molecules.csv", index=False)
    table.to_csv(args.out / "test_metrics_by_group.csv", index=False)
    metrics = {
        "experiment": "Fine-tuned DreaMS (100% split, best epoch) on the held-out test set, scored once",
        "checkpoint": str(args.checkpoint), "test_set": str(args.test),
        "notes": {
            "closed_set": "DreaMS predicts only training molecules; test molecules without training spectra are misses",
            "rank": "rank of the true molecule in the top 25, 0 if not there",
            "molecule_scores": "each candidate scored by its highest probability over the molecule's spectra",
        },
        "self_check": self_check,
        "by_group": table.to_dict(orient="records"),
        "seconds": time.time() - started, "gpu": gpu,
        "pytorch_version": torch.__version__, "python_version": platform.python_version(),
    }
    (args.out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(table[["group", "spectra", "molecules", "molecules_with_training_spectra", "spectrum_top1",
                     "spectrum_top25", "molecule_top1", "molecule_mrr25"]])
    print(f"written to {args.out} in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
