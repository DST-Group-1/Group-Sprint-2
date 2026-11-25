# Running the Sprint 2 benchmarks on BluePebble

This folder runs the group's notebooks on the University of Bristol's BluePebble cluster as Slurm
batch jobs: the cleaned data is rebuilt from the Kaggle download, Michael's notebooks build a 100%
split from it, and each of the three models runs on that full training set. The 5% and 9% results
already in the repo (from the RTX 4080) stay as they are. No data is copied from a laptop: the
cluster regenerates it with `LukeD/01`, and job 01 checks it matches the laptop run count for count.

The notebooks themselves are not edited. Each job writes a copy next to the original with a few
settings changed (paths, epochs, worker counts) using `nbparam.py`, runs the copy, and keeps it with
its outputs as the record of the run (`*-bluepebble*.ipynb`).

## What runs

| Step | Job | Runs | Resources | Rough time | Small results (to commit) |
| --- | --- | --- | --- | --- | --- |
| smoke | `00_smoke_gpu` | a GPU check in each Python environment | 1 GPU | minutes | log only |
| data | `01_data` | `LukeD/01-data-and-validation` as it is | 8 cores, 32 GB | 10-15 min | `LukeD/01-...-bluepebble.ipynb` |
| scaling | `02_loading_scaling` | `LukeD/02-loading-and-scaling`, 1 to 32 workers (optional: `bash submit_all.sh` without `scaling` skips it) | a whole node | about 1 h | `LukeD/benchmark_results/bluepebble/` |
| splits | `03_group_splits` | Michael's split notebooks: `stage_1_5pct` (it defines the shared validation set), then `stage_3_100pct` | 160 GB | 30-60 min | the executed notebooks |
| mlp | `04_mlp` | `ThomasF/03a-fingerprint-mlp`, section 13 on the 100% split, reusing his saved runs | 1 GPU | about 1 h | `ThomasF/results_bluepebble/` |
| ms2ds | `05_ms2deepscore` | `DavidA/MS2DeepScore` in its 100% mode | 1 GPU | about 40 min | `DavidA/results_bluepebble/100pct/` |
| dreams | `06_dreams` | `MichaelM/DreaMS_tuned_scaled_executed` on the 100% split, 3 epochs | 1 GPU | 8-10 h | `MichaelM/results_bluepebble/` |

`submit_all.sh` chains them: data, then scaling and splits; the three model jobs wait for the
splits and for the smoke test. To skip the scaling benchmark, name the other steps:
`bash submit_all.sh smoke data splits mlp ms2ds dreams`. Model weights, embeddings and the splits stay on work storage.

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
3. **Check `config.sh`.** It is filled in from the `sacctmgr` and `sinfo` output of 25 Nov: project
   `math002941`; CPU jobs on `short,compute`; GPU jobs that fit in 6 hours on `gpu_short,gpu` (Slurm
   starts them in whichever frees up first) and the longer ones on `gpu`; every GPU job on an RTX
   3090, so all three 100% runs used the same kind of card. Only change it if `echo $WORK` is empty
   (see step 2).
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
7. **Watch**: `squeue --me` (and `squeue --me --start` for Slurm's estimate of when pending jobs
   start: every node was busy on 25 Nov, so expect to queue); logs are in `logs/`, e.g. `tail -f logs/casmi-04-mlp-<jobid>.out`;
   `sacct --starttime today` shows finished jobs; `scancel <jobid>` stops one.
8. **Bring the results back**: `bash collect.sh` writes `logs/sacct.csv` (wall time, peak memory and
   allocation of every job, for the compute table in the report) and lists the small result files.
   Commit those and push, or `rsync` them to a laptop.

For the markers: the report must say where a manual step is needed (the brief mentions waiting for a
cluster job). These runs are reproduced by steps 1 to 8; the results the report reads are committed.

## Things that matter for the report

- **The 100% times come from a different card.** The 5% and 9% runs were on an RTX 4080, the 100%
  runs on an RTX 3090, so a time-against-data curve mixes two devices. Job 04 helps here: Thomas's
  section 10 times the same MLP on each card it runs on, so `compute.json` ends up with a measured
  4080-to-3090 speed ratio to put the times on one scale (for the MLP's workload; DreaMS's may
  differ a little). The MLP also has a same-card series already: Thomas's 1% to 100% runs, all on
  the 4080.
- **The validation set grows with the training set.** Every strand scores only validation spectra
  whose molecule is in training: 10,153 at 5%, 13,550 at 9%, and more at 100%. Accuracy across
  fractions is only comparable on a fixed set, the spectra known at 5% (as in
  `ThomasF/comparison/accuracy_5pct_vs_9pct_same_spectra.csv`). The MS2DeepScore job already scores
  that set; for DreaMS and the MLP, filter their 100% `validation_predictions.csv` to it.
- **DreaMS at 100% is where the cost shows.** At 9% an epoch took 567 s for 179,108 spectra on the
  4080, so the 2.27 million spectra of the full split take about 2 h per epoch there. Job 06 stops
  at 3 epochs (`DREAMS_MAX_EPOCHS`). That is still 6.8 million training examples, more than the 3.6
  million of the 9% run's 20 epochs, and its per-epoch history compares with epochs 1 to 3 of the
  9% run. At 1000 times the data the same arithmetic gives about 80 days per epoch on one card:
  that is the case for data-parallel training across GPUs.
- **`os.cpu_count()` reports the whole node**, not what Slurm allocated. Job 02 takes a whole node
  for that reason, which also keeps other jobs out of its timings. The MLP job runs on part of a GPU
  node, so in Thomas's section 10 the CPU thread counts above the 4 allocated cores share those 4
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
| `no kernel image is available` in the smoke test | that torch build does not support the card: keep `GPU_GRES` on an RTX type |
| The 3090s stay busy for a day | switch `GPU_GRES` to `gpu:rtx_2080:1` (11 GB; DreaMS at 100% should fit, just) or `gpu:a100:1` (gpu partition only; set `GPU_SHORT_PARTITION=gpu` too), `scancel` the waiting jobs and resubmit them by name |
| `$'\r': command not found` | Windows line endings, from pasting or a Windows checkout: `sed -i 's/\r$//' <file>` |
| A notebook cell fails | the traceback is in the log; the original notebook is untouched, so fix and resubmit the step |
| Job 01 says the counts differ | the Kaggle file or a package version changed; the log prints both sets of counts |

Partitions, card types and quotas are from BluePebble's own `sinfo`, `sacctmgr` and `user-quota` on
25 Nov 2026. Not checked: whether compute nodes reach the internet (so everything is downloaded
during setup), and how many cores and how much memory each GPU node has; the GPU jobs ask for 4
cores and 40 GB, which `sinfo -p gpu -N -o "%N %G %c %m"` can confirm.
