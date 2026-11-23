# Results on the group's 5% split (stage_1_5pct)

Made by run_gpu_jobs.py on 23 Nov 2026 12:17. Validation = the spectra whose molecule is in the training split (split by spectrum, so other spectra of the same molecules are in training).

| Model | Spectra | Top-1 | Top-5 | Balanced acc. | Macro F1 | Weighted F1 | Epochs |
|---|---|---|---|---|---|---|---|
| Frozen DreaMS, head only (Michael) | 10,153 | 0.327 | – | 0.290 | 0.279 | 0.320 | 20, best kept |
| Fine-tuned DreaMS, all layers (Michael) | 10,153 | 0.406 | – | 0.374 | 0.357 | 0.390 | 20 of 20, best kept |
| CNN (David) | 10,153 | 0.001 | 0.005 | – | – | – | 8, last kept |
| Fingerprint MLP (Thomas), all training molecules | 10,153 | 0.279 | 0.400 | 0.247 | 0.236 | – | 14 of 18, dev |
| Fingerprint MLP, candidates within ±10 ppm | 10,153 | 0.626 | 0.827 | 0.624 | 0.620 | – | 14 of 18, dev |

Notes:

- Frozen and fine-tuned DreaMS scored the same validation spectra (10,153 and 10,153).
- Top-1 is accuracy. Balanced accuracy and macro F1 are averaged over the validation molecules and recomputed here from each DreaMS run's saved predictions, so every row uses one definition; the fine-tuned notebook's own macro F1 (0.255) also averages in molecules that were predicted but never the answer, which pulls it down.
- Both DreaMS runs keep the epoch with the best accuracy on this validation set, and the fine-tuned run also stops early on it, so their scores are slightly optimistic. The CNN reports its last epoch; the MLP stops on its own dev molecules.
- Fine-tuning: encoder lr 1e-05, head lr 0.0003, weight decay 0.01, dropout 0.3, batch 32, up to 20 epochs (patience 5); 1.53 h on NVIDIA GeForce RTX 4080.
- MLP results read from D:\Data Science Toolbox\Group Sprints\Group-Sprint-2\results\mlp\group_5pct\summary.json.
