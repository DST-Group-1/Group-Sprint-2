# Running the Sprint 2 benchmarks on BluePebble

This folder runs the group's notebooks on the University of Bristol's BluePebble cluster as Slurm
batch jobs: the cleaned data and Michael's splits are rebuilt from the Kaggle download, then each
model runs on the 5%, 9% and full (100%) training sets. No data is copied from a laptop: the cluster
regenerates it with `LukeD/01`, and job 01 checks it matches the laptop run count for count.

The notebooks themselves are not edited. Each job writes a copy next to the original with a few
settings changed (paths, epochs, worker counts) using `nbparam.py`, runs the copy, and keeps it with
its outputs as the record of the run (`*-bluepebble*.ipynb`).

## What runs

| Step | Job | Runs | Resources | Rough time | Small results (to commit) |
| --- | --- | --- | --- | --- | --- |
| smoke | `00_smoke_gpu` | a GPU check in each Python environment | 1 GPU | minutes | log only |
| data | `01_data` | `LukeD/01-data-and-validation` as it is | 8 cores, 32 GB | minutes | `LukeD/01-...-bluepebble.ipynb` |
| scaling | `02_loading_scaling` | `LukeD/02-loading-and-scaling`, 1 to 32 workers | a whole node | about 1 h | `LukeD/benchmark_results/bluepebble/` |
| splits | `03_group_splits` | Michael's two split notebooks: `stage_1_5pct`, `stage_2_9pct`, and a new `stage_3_100pct` | 160 GB | under 1 h expected | the executed notebooks |
| mlp | `04_mlp` | `ThomasF/03a-fingerprint-mlp`, "full" plan, section 13 on each split | 1 GPU | 2 to 5 h first pass (Thomas's estimate), then short passes | `ThomasF/results_bluepebble/` |
| ms2ds | `05_ms2deepscore` | `DavidA/MS2DeepScore` with 5%, 9% and 100% reference sets | 1 GPU per task | the 9% run took 3 min on the 4080, so 100% about 40 min | `DavidA/results_bluepebble/` |
| dreams | `06_dreams` | `MichaelM/DreaMS_tuned_scaled_executed` on each split, 3 epochs | 1 GPU per task | 100%: about 2 h per epoch on the 4080, longer here | `MichaelM/results_bluepebble/` |

`submit_all.sh` chains them: data, then scaling and splits; the three model jobs wait for the
splits and for the smoke test. Model weights, embeddings and the splits stay on work storage.

## Before you start, on your laptop

1. Commit and push this folder, so the clone on BluePebble has it.
2. Add these lines to the repo's `.gitignore`. On BluePebble the 100% split, model weights and
   embeddings sit inside or next to the repo, and none of them should be committed:
   ```
   .venv/
   model_datasets/
   *.pt
   *.ckpt
   *.npy
   ```
3. Ask David for the pretrained `ms2deepscore_model.pt` his notebook uses, and where it came from
   (the report should cite it). The DreaMS weights are downloaded by the setup script.

## On BluePebble

1. **Log in** from the university network or the VPN: `ssh <username>@bp1-login.acrc.bris.ac.uk`.
   Then look up four things:
   ```
   sacctmgr show user withassoc format=account where user=$USER   # your project code
   sinfo -s                                                         # partitions and time limits
   sinfo -o "%20P %40G %l"                                          # GPU types per partition
   user-quota                                                       # home and work storage
   ```
2. **Clone the repo onto work storage** (`$HOME` is only about 20 GB):
   ```
   mkdir -p $WORK/casmi26 && cd $WORK/casmi26
   git clone <the DST-Group-1/Group-Sprint-2 URL> Group-Sprint-2
   cd Group-Sprint-2/LukeD/bluepebble
   ```
   If `$WORK` is not set, use the work path `user-quota` shows: `export CASMI_WORK=<that path>/casmi26`
   before the commands, and add the same line to `~/.bashrc`.
3. **Edit `config.sh`**: the project code, the two partition names, and `GPU_GRES`. Pin one card
   type (for example `gpu:rtx_3090:1`, with the name `sinfo` showed), so every point of a scaling
   curve ran on the same hardware.
4. **Credentials and weights.** Copy your Kaggle API token to `~/.kaggle/` on BluePebble
   (`scp` it from your laptop, then `chmod 600` it), and copy David's model to
   `$WORK/casmi26/models/ms2deepscore_model.pt`.
5. **Set up**, on the login node, inside `tmux` (about half an hour):
   ```
   tmux new -s setup
   bash setup_login_node.sh
   ```
   This installs Miniforge and three environments on work storage (`casmi` for Luke, Thomas and the
   splits; `dreams`; `ms2ds`, because the strands pin incompatible versions), downloads the
   competition with `kagglehub`, and downloads and checksums the DreaMS checkpoint. Everything that
   needs the internet happens here, as compute nodes may not have it.
6. **Submit**: `bash submit_all.sh`. Or one step at a time, such as `bash submit_all.sh smoke`, and the
   rest once it passes. Steps can be rerun by name: `bash submit_all.sh mlp`.
7. **Watch**: `squeue --me`; logs are in `logs/`, e.g. `tail -f logs/casmi-04-mlp-<jobid>.out`;
   `sacct --starttime today` shows finished jobs; `scancel <jobid>` stops one.
8. **Bring the results back**: `bash collect.sh` writes `logs/sacct.csv` (wall time, peak memory and
   allocation of every job, for the compute table in the report) and lists the small result files.
   Commit those and push, or `rsync` them to a laptop.

For the markers: the report must say where a manual step is needed (the brief mentions waiting for a
cluster job). These runs are reproduced by steps 1 to 8; the results the report reads are committed.

## Things that matter for the report

- **Compare times on one device.** The 5% and 9% results already in the repo came from an RTX 4080.
  Job 04 retrains the whole MLP series on the pinned BluePebble card (into a new folder), and job 06
  reruns DreaMS at 5% and 9%, so each curve has every point on one card. Thomas's `compute.json`
  keeps each device separately.
- **The validation set grows with the training set.** Every strand scores only validation spectra
  whose molecule is in training: 10,153 at 5%, 13,550 at 9%, and more at 100%. Accuracy across
  fractions is only comparable on a fixed set, the spectra known at 5% (as in
  `ThomasF/comparison/accuracy_5pct_vs_9pct_same_spectra.csv`). MS2DeepScore tasks 0 and 2 already
  share that set; for DreaMS and the MLP, filter `validation_predictions.csv` to it.
- **DreaMS at 100% is where the cost shows.** At 9% an epoch took 567 s for 179,108 spectra on the
  4080, so the 2.27 million spectra of the full split take about 2 h per epoch there, and the 20
  epochs of the 9% run would take about 40 h. Job 06 stops at 3 epochs (`DREAMS_MAX_EPOCHS`), which
  measures seconds per epoch against data size and accuracy at equal epochs. At 1000 times the data
  the same arithmetic gives about 80 days per epoch on one card: that is the case for data-parallel
  training across GPUs.
- **`os.cpu_count()` reports the whole node**, not what Slurm allocated. Job 02 takes a whole node
  for that reason, which also keeps other jobs out of its timings. The MLP job runs on part of a GPU
  node, so in Thomas's section 10 the CPU thread counts above the 8 allocated cores share those 8
  cores and should not be read as scaling (`os.process_cpu_count()` in Python 3.13 and later counts
  only the usable cores).
- **Storage is a 1000× constraint too.** The raw training file is 3 GB, so 1000 times is about 3 TB,
  more than a typical work quota (check `user-quota`). At that scale the data would live on shared
  research storage and be streamed, which is a concrete point for the 1000× discussion.
- **Loading runs read from the cluster's shared filesystem**, not a local SSD as on a laptop. That
  is closer to a real cluster, and may make the loading curves of job 02 differ from the laptop's.
- If there is time: a BluePebble GPU node has several cards, so a 1, 2 and 4 GPU data-parallel
  throughput test of the MLP would turn the all-reduce estimate in Thomas's section 10.2 into a
  measurement.

## If something goes wrong

| Symptom | What to do |
| --- | --- |
| A job stays pending | `squeue --me -o "%i %j %T %r"` shows why. `DependencyNeverSatisfied` means an earlier step failed: read its log, `scancel` the waiting jobs, fix it, resubmit those steps by name |
| `OUT_OF_MEMORY` in `sacct` | raise `--mem` at the top of that job script |
| `TIMEOUT` | raise `--time`, within the partition's limit from `sinfo -s` |
| `no kernel image is available` in the smoke test | that torch build does not support the card: pin `GPU_GRES` to an RTX card type |
| A notebook cell fails | the traceback is in the log; the original notebook is untouched, so fix and resubmit the step |
| Job 01 says the counts differ | the Kaggle file or a package version changed; the log prints both sets of counts |

What this could not check from outside: the exact partition and GPU type names, their time limits,
and whether compute nodes reach the internet. The ACRC documentation is behind the university
login, so step 1 looks them up, and the defaults in `config.sh` (`compute`, `gpu`) may need changing.
