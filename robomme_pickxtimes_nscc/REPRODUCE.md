# Reproduction Notes

These notes assume the official RoboMME dataset, official MME-VLA checkpoint, and NSCC environment are already prepared.

## Data View

PickXtimes uses the task block `episodes 500-599`, which produced `100 episodes` and `53,720 execution samples`.

```bash
python scripts/nscc_build_views_by_episode.py \
  --source /scratch/users/nus/<NSCC_USER>/robomme_nscc/data/robomme_preprocessed_data \
  --output-root /scratch/users/nus/<NSCC_USER>/robomme_nscc/data \
  --suffix taskfull \
  --slugs pickxtimes
```

## Main Experiment

The main PBS job runs:

1. Official 512-budget eval.
2. 2048-budget official wrapper eval.
3. 512-budget PickXtimes specialist train/eval.
4. 2048-budget PickXtimes specialist train/eval.

```bash
qsub pbs/pbs_gpu_pickxtimes_budget2048.pbs
```

The first attempted official 2048 wrapper had a trailing-newline issue in `history_config.txt`. This was fixed by stripping `history_config.txt` on read and writing wrapper config names without a newline. The final official 2048 result comes from the dedicated follow-up job:

```bash
qsub pbs/pbs_gpu_pickxtimes_official2048_eval.pbs
```

## Alignment Fixes

The important fix was to copy norm stats from the official checkpoint:

```bash
OFFICIAL_NORM_STATS=/scratch/users/nus/<NSCC_USER>/robomme_nscc/runs/official/perceptual-framesamp-modul/79999/assets/robomme/norm_stats.json
```

Without this fix, the earlier PickXtimes specialist reached only `41/50`. With the fix, the 512-budget specialist reaches `44/50`, matching the official checkpoint.

## Non-Aligned Items

This is not a full official retrain. It is a task-specialist adaptation:

| Item | Official | This study |
|---|---|---|
| Dataset | all tasks | PickXtimes-only |
| Steps | 80k | 500 |
| Batch | 64 | 8 for 512, 2 for 2048 |
| Devices | 4 FSDP | 1 GPU |
| Trainable params | official MME-VLA setup | `mem|action|time` only |

These differences are intentional to support rapid challenge iteration.
