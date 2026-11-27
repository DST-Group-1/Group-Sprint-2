## 15. The held-out test set, scored once (added by LukeD/bluepebble/jobs/09_mlp_test.sbatch)

This section is not in the original notebook: job 09 appends it to a copy and runs the copy. No
model is trained in it. Both models are the ones job 04 trained, loaded from their saved weights,
and every choice they use (settings, scoring rule, pooling, mass window) was fixed on dev before
this point. Before it scores the test set, each model must reproduce job 04's validation results;
otherwise the section stops.

* **Final model** (section 12), under this notebook's protocol: one ranked list per test molecule,
  candidates within ±10 ppm from `molecules.in_candidates` (class 3 structures withheld).
* **Group-split model** (section 13), the one in the comparison with DreaMS and MS2DeepScore, with
  the test spectra of 01's held-out test molecules (the same spectra as the other strands' test jobs).
  Candidates are either the training split's molecules (the setting DreaMS and MS2DeepScore are
  limited to) or those plus the test molecules, class 3 withheld. Scores are given per spectrum and
  per molecule, a molecule's spectra pooled by each candidate's best score over them.
