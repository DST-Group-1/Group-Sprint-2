# Appended by LukeD/bluepebble/jobs/09_mlp_test.sbatch (see the markdown cell above). No training here.
TEST_OUT = Path(os.environ["MLP_TEST_OUT"])                # small result files, meant to be committed
REFERENCE = Path(os.environ["MLP_REFERENCE_DIR"])          # job 04's results folder: read, never written
TEST_SET = Path(os.environ["TEST_DIR"])                    # job 07's test set, shared with the other strands
CHECK_TOL = 0.002                                          # validation MRR may move this much (GPU arithmetic)
TEST_OUT.mkdir(parents=True, exist_ok=True)
assert GROUP_DIR is not None and GROUP_DIR.name == "stage_3_100pct", f"expected the 100% group split, got {GROUP_DIR}"

# The same test spectra as job 07's test set, which the DreaMS and MS2DeepScore test jobs score.
shared_ids = set(pl.read_parquet(TEST_SET / "test.parquet", columns=["row_id"])["row_id"].cast(pl.Int64).to_list())
assert set(TEST.spectra.row_id.tolist()) == shared_ids, "this notebook's test spectra differ from job 07's test set"
TEST_GROUPS = {"all": None, "class 1": pl.col("sim_class") == 1, "class 2": pl.col("sim_class") == 2,
               "class 3": pl.col("sim_class") == 3, "natural-product examples": pl.col("pool") == NP,
               "molecules with training spectra": pl.col("in_training")}

# ---- 1. The final model (section 12), under this notebook's protocol ---------------------------------------------
reference_final = json.loads((REFERENCE / "final" / "summary.json").read_text())
final_check = {g: (FINAL_VAL[g]["mrr"], reference_final["val"][g]["mrr"]) for g in EVAL_GROUPS}
print("final model, validation MRR@25 now and in job 04:", {g: f"{a:.4f} / {b:.4f}" for g, (a, b) in final_check.items()})
if not all(abs(a - b) <= CHECK_TOL for a, b in final_check.values()):
    raise RuntimeError("the final model does not reproduce job 04's validation results, so the test set was NOT scored")
final_test = rank_queries(TEST, model_score(TEST, predict(final.model, TEST.spectra, FINAL),
                                            BEST_SCORING["scoring"], BEST_SCORING["aggregate"]))
FINAL_TEST = summarise(final_test)
c3 = final_test.filter(pl.col("sim_class") == 3)
FINAL_TEST["class 3"].update(   # nanmean: a molecule with no candidate in its window has no top-1
    top1_tanimoto=float(np.nanmean(c3["top1_tanimoto"].to_numpy())) if c3["top1_tanimoto"].is_not_nan().any() else math.nan,
    molecules_with_candidates=int((c3["n_candidates"] > 0).sum()))
final_test.write_parquet(TEST_OUT / "final_test_molecules.parquet")
print("final model on the test set (per molecule, ±10 ppm candidates, class 3 withheld):")
show(summary_table({"final MLP, test": FINAL_TEST, "final MLP, validation": FINAL_VAL}))
print("class 3, mean Tanimoto of the top-1 guess to the answer:", FINAL_TEST["class 3"])


# ---- 2. The group-split model (section 13) ----------------------------------------------------------------------
def rank_molecules_max(logits, truth_mol, mass, cand, ppm, scoring, chunk=512):
    # One ranked list per molecule: each candidate scored by its best score over the molecule's spectra, the
    # pooling the DreaMS and MS2DeepScore test jobs use. Scores, windows and ties exactly as in rank_spectra;
    # rows must be grouped by molecule.
    uniq, inverse = np.unique(FP[cand], axis=0, return_inverse=True)
    Fm = torch.from_numpy(np.unpackbits(uniq, axis=1).astype(np.float32)).to(DEVICE)
    n_set, inv = Fm.sum(1), torch.from_numpy(inverse.reshape(-1)).to(DEVICE)
    by_mass = np.argsort(MONO_MASS[cand], kind="stable")
    mass_pos = torch.from_numpy(np.argsort(by_mass)).to(DEVICE)
    sorted_mass = MONO_MASS[cand][by_mass]
    lo = np.searchsorted(sorted_mass, mass * (1 - (ppm or 0) * 1e-6), "left")
    hi = np.searchsorted(sorted_mass, mass * (1 + (ppm or 0) * 1e-6), "right")
    lo, hi = np.where(np.isnan(mass), 0, lo), np.where(np.isnan(mass), 0, hi)
    bounds = np.flatnonzero(np.diff(truth_mol)) + 1
    starts, ends = np.concatenate([[0], bounds]), np.concatenate([bounds, [len(truth_mol)]])
    mols = truth_mol[starts]
    at = np.searchsorted(cand, mols)
    truth = np.where((at < len(cand)) & (cand[np.minimum(at, len(cand) - 1)] == mols), at, -1)
    n = len(mols)
    g, t, n_cand, has = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, bool)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ends[j + 1] - starts[i] <= chunk:
            j += 1
        rows = slice(starts[i], ends[j])
        z = torch.from_numpy(logits[rows]).to(DEVICE)
        p = torch.sigmoid(z)
        if scoring == "loglik":
            s = z @ Fm.T
        else:
            inter = p @ Fm.T
            s = (inter / (p.sum(1, keepdim=True) + n_set - inter).clamp_min(1e-9) if scoring == "tanimoto"
                 else inter / (p.norm(dim=1, keepdim=True) * n_set.sqrt() + 1e-9))
        s = s[:, inv]
        if ppm is not None:
            inside = ((mass_pos[None, :] >= torch.from_numpy(lo[rows]).to(DEVICE)[:, None])
                      & (mass_pos[None, :] < torch.from_numpy(hi[rows]).to(DEVICE)[:, None]))
            s = s.masked_fill(~inside, -torch.inf)
        for m in range(i, j + 1):
            pooled = s[starts[m] - starts[i]:ends[m] - starts[i]].max(0).values
            n_cand[m] = int(torch.isfinite(pooled).sum())
            if truth[m] >= 0 and torch.isfinite(pooled[truth[m]]):
                answer = pooled[truth[m]]
                g[m], t[m], has[m] = int((pooled > answer).sum()), int((pooled == answer).sum()) - 1, True
        i = j + 1
    last, first = np.minimum(g + t + 1, TOP_K), np.minimum(g + 1, TOP_K + 1)
    rr = np.where(has & (g + 1 <= TOP_K), (HARMONIC[last] - HARMONIC[first - 1]) / (t + 1), 0.0)
    hits = {f"hit{k}": np.where(has, np.clip(np.minimum(g + t + 1, k) - g, 0, None) / (t + 1), 0.0) for k in (1, 5, 25)}
    return pl.DataFrame({"mol": mols, "n_candidates": n_cand, "truth_in_set": has, "rr": rr, **hits})


weights = CACHE_DIR / "models" / f"{G_RESULTS.name}_{G_SUMMARY['key']}.pt"
assert weights.exists(), f"no saved weights for the group-split model at {weights}"
g_model = build_model(GROUP_CFG, np.full(FP_BITS, 0.5))
g_model.load_state_dict(torch.load(weights, map_location=DEVICE))
g_model.eval()

# Self-check: job 04's validation scores for this model, recomputed from the loaded weights.
g_logits = predict(g_model, GROUP_VAL, GROUP_CFG)
check_name, check_mask, check_cand, _ = G_SETTINGS[0]          # known molecules, all training molecules
sel = np.flatnonzero(check_mask)
again = rank_spectra(g_logits[sel], GROUP_VAL.mol[sel].astype(np.int64), G_MASS[sel], check_cand, None, G_SCORING["all"])
saved = pl.read_parquet(REFERENCE / G_RESULTS.name / "val_spectra.parquet").filter(pl.col("setting") == check_name)
paired = again.select("rr", "hit1").with_columns(pl.Series("row_id", GROUP_VAL.row_id[sel])).join(
    saved.select("row_id", "rr", "hit1"), on="row_id", how="inner", suffix="_saved")
G_CHECK = {"setting": check_name, "spectra": len(sel), "saved_spectra": saved.height, "paired": paired.height,
           "hit1_agreement": float(((paired["hit1"] - paired["hit1_saved"]).abs() < 1e-6).mean()),
           "mrr_now": float(again["rr"].mean()), "mrr_saved": float(saved["rr"].mean()),
           "top1_now": float(again["hit1"].mean()), "top1_saved": float(saved["hit1"].mean())}
print("group-split model, self-check on job 04's validation spectra:", G_CHECK)
if not (paired.height == len(sel) == saved.height and G_CHECK["hit1_agreement"] >= 0.995
        and abs(G_CHECK["mrr_now"] - G_CHECK["mrr_saved"]) <= CHECK_TOL):
    raise RuntimeError("the group-split model does not reproduce job 04's validation results, so the test set was NOT scored")
del g_logits, again

# The test spectra, grouped by molecule.
t_frame = (heldout.filter(pl.col("split") == "test").select(SPECTRUM_COLUMNS).filter(pl.col("mzs").list.len() > 0)
           .join(molecules.select("inchikey14", "mol", "sim_class", "pool", "in_candidates"), on="inchikey14", how="inner")
           .with_columns(pl.lit(0.0).alias("train_subset_rank")).sort("mol", "row_id"))
assert set(t_frame["row_id"].cast(pl.Int64).to_list()) == shared_ids
T_SS = load_spectra(t_frame)
T_MOL = T_SS.mol.astype(np.int64)
T_MASS = neutral_masses(t_frame["precursor_mz"].to_numpy(), t_frame["adduct"].to_list())
key_of = np.full(int(molecules["mol"].max()) + 1, "", dtype=object)
key_of[molecules["mol"].to_numpy()] = molecules["inchikey14"].to_numpy()
spectrum_info = t_frame.select("row_id", "inchikey14", "sim_class", "pool").with_columns(
    pl.Series("in_training", np.isin(T_MOL, CLASS_MOLS)))
molecule_info = spectrum_info.group_by("inchikey14", maintain_order=True).agg(
    pl.col("sim_class").first(), pl.col("pool").first(), pl.col("in_training").first(), pl.len().alias("n_spectra"))
OPEN_TEST = np.union1d(CLASS_MOLS, np.unique(T_MOL[t_frame["in_candidates"].to_numpy()])).astype(np.int64)
T_SETTINGS = [("training molecules", CLASS_MOLS, None),
              (f"training molecules within ±{PPM_WINDOW:g} ppm", CLASS_MOLS, PPM_WINDOW),
              ("training + test molecules, class 3 withheld", OPEN_TEST, None),
              (f"training + test molecules, class 3 withheld, within ±{PPM_WINDOW:g} ppm", OPEN_TEST, PPM_WINDOW)]

t_logits = predict(g_model, T_SS, GROUP_CFG)
spectrum_frames, molecule_frames, table_rows = [], [], []
for name, cand, ppm in T_SETTINGS:
    scoring = G_SCORING["all" if ppm is None else "window"]
    res = rank_spectra(t_logits, T_MOL, T_MASS, cand, ppm, scoring)
    guess = res["guess_mol"].to_numpy()
    res = pl.concat([spectrum_info, res.select("n_candidates", "truth_in_set", "rr", "hit1", "hit5", "hit25"),
                     pl.DataFrame({"guess_inchikey14": np.where(guess >= 0, key_of[np.maximum(guess, 0)], "").astype(str)})],
                    how="horizontal").with_columns(pl.lit(name).alias("setting"), pl.lit(scoring).alias("scoring"))
    pooled = pl.concat([molecule_info, rank_molecules_max(t_logits, T_MOL, T_MASS, cand, ppm, scoring).drop("mol")],
                       how="horizontal").with_columns(pl.lit(name).alias("setting"))
    spectrum_frames.append(res)
    molecule_frames.append(pooled)
    for group, cond in TEST_GROUPS.items():
        s = res if cond is None else res.filter(cond)
        m = pooled if cond is None else pooled.filter(cond)
        if not s.height:
            continue
        table_rows.append({
            "setting": name, "scoring": scoring, "group": group, "spectra": s.height, "molecules": m.height,
            "molecules_with_training_spectra": int(m["in_training"].sum()),
            "spectrum_top1": float(s["hit1"].mean()), "spectrum_top5": float(s["hit5"].mean()),
            "spectrum_top25": float(s["hit25"].mean()), "spectrum_mrr25": float(s["rr"].mean()),
            "spectrum_top1_molecule_averaged": float(s.group_by("inchikey14").agg(pl.col("hit1").mean())["hit1"].mean()),
            "molecule_top1": float(m["hit1"].mean()), "molecule_top5": float(m["hit5"].mean()),
            "molecule_top25": float(m["hit25"].mean()), "molecule_mrr25": float(m["rr"].mean()),
            "answer_among_candidates": float(s["truth_in_set"].mean())})
G_TEST_TABLE = pl.DataFrame(table_rows)
pl.concat(spectrum_frames).write_parquet(TEST_OUT / "group_test_spectra.parquet")
pl.concat(molecule_frames).write_parquet(TEST_OUT / "group_test_molecules.parquet")
G_TEST_TABLE.write_csv(TEST_OUT / "group_test_metrics_by_group.csv")
(TEST_OUT / "test_summary.json").write_text(json.dumps({
    "test_set": str(TEST_SET), "test_spectra": T_SS.n, "test_molecules": molecule_info.height,
    "final_model": {"config": asdict(FINAL), "scoring": BEST_SCORING["scoring"], "aggregate": BEST_SCORING["aggregate"],
                    "self_check_val_mrr": {g: {"now": a, "job_04": b} for g, (a, b) in final_check.items()},
                    "test": FINAL_TEST},
    "group_model": {"split": GROUP_DIR.name, "weights": str(weights), "config": asdict(GROUP_CFG), "scoring": G_SCORING,
                    "self_check": G_CHECK, "test": table_rows},
    "notes": {"no_training": "both models load job 04's weights; no setting is chosen on the test set",
              "ties": "hits and reciprocal ranks are expectations under random tie-breaking, as in section 3.4",
              "molecule_scores": "group model: each candidate's best score over the molecule's spectra"},
    "device": DEVICE_NAME, "torch": torch.__version__, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, indent=2, default=float))
show(G_TEST_TABLE.filter(pl.col("group").is_in(["all", "class 1", "class 2", "class 3", "molecules with training spectra"]))
     .select("setting", "group", "spectra", "molecules", "spectrum_top1", "spectrum_top25", "molecule_top1", "molecule_mrr25"))
print("written:", *sorted(p.name for p in TEST_OUT.iterdir()))
