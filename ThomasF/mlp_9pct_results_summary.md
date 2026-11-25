# Fingerprint MLP (Thomas): results on the 9% split (stage_2_9pct)

Made by run_mlp_9pct.py on 24 Nov 2026 17:36.

Trained on the split's 179,108 training spectra (leak-checked: no validation spectrum in training, none twice), scored one validation spectrum at a time with the same metrics as the CNN, DreaMS and MS2DeepScore runs. Candidates are the training molecules (the classifiers' label set), all of them or only those within ±10 ppm of the spectrum's mass. Validation spectra of molecules outside training are scored with the validation molecules added to the candidates.

| Validation spectra | Candidates | Spectra | Top-1 | Top-5 | MRR@25 | Balanced acc. | Macro F1 |
|---|---|---|---|---|---|---|---|
| known molecules | all training molecules | 13,550 | 0.305 | 0.425 | 0.360 | 0.268 | 0.258 |
| known molecules | training molecules within ±10 ppm | 13,550 | 0.623 | 0.811 | 0.707 | 0.617 | 0.614 |
| unseen molecules | training + validation molecules | 9,413 | 0.022 | 0.092 | 0.057 | 0.022 | 0.021 |
| unseen molecules | training + validation molecules within ±10 ppm | 9,413 | 0.443 | 0.718 | 0.568 | 0.445 | 0.441 |

Training: 178,610 spectra (dev molecules left out), 24 epochs (kept 23, chosen on the notebook's own dev molecules, not on this validation set), 1.6 min on NVIDIA GeForce RTX 4080; settings as the final model, width 2048.

The folders results\fingerprint_mlp_9pct and results\fingerprint_mlp_10ppm_9pct hold the same files and graphs as fine-tuned DreaMS's results folder (copies in gpu_run_package\all_results).

