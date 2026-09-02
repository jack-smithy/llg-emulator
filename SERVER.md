# gto4 — GPU server: setup and usage guidelines

Status: 2026-08-24

gto4 is the group's new GPU compute server. This document describes what the
machine is, how to get work onto it, and the few rules that keep it usable
for everyone.

---

## 1. The machine

|           |                                                             |
| --------- | ----------------------------------------------------------- |
| CPU       | 2× AMD EPYC 9575F — 128 cores (no hyper-threading)          |
| RAM       | 377 GiB + 192 GB NVMe swap                                  |
| GPU       | 5× NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB each |
| Disks     | `/` 877 GB (NVMe), `/data` 3.5 TB (NVMe)                    |
| OS        | Ubuntu 24.04, NVIDIA driver 595                             |
| Scheduler | SLURM 23.11, single node, partition `main`                  |

The five GPUs are not all offered the same way. Two of them are split into
four hardware-isolated 24 GB partitions each (NVIDIA MIG), the other three
are handed out whole:

| You ask for                 | You get                                 | How many exist |
| --------------------------- | --------------------------------------- | -------------- |
| `--gres=gpu:1g.24gb:1`      | one isolated 24 GB MIG slice            | 8              |
| `--gres=gpu:full:1`         | one whole 96 GB GPU, exclusively        | 3              |
| `--gres=gpu:full:2` or `:3` | several whole GPUs (multi-GPU training) | —              |

MIG isolation is enforced in hardware: someone else's job on a neighbouring
slice of the same physical GPU cannot touch your memory, slow you down, or
crash you.

**Slices do not add up.** Two 24 GB slices in one job are *not* 48 GB — CUDA
uses one MIG instance per process. If your model does not fit in 24 GB, ask
for `gpu:full`.

---

## 2. The one rule that matters

**All GPU work goes through SLURM.** Running CUDA directly from an SSH shell
does not work: the GPU device files are root-only, and SLURM grants your job
access to exactly the devices it allocated, for exactly as long as it runs.

This is deliberate. It is what stops two jobs from landing on the same GPU
and OOM-killing each other, and it is what makes the queue fair.

Inside a job, `nvidia-smi` and CUDA see exactly your allocation. **Do not set
`CUDA_VISIBLE_DEVICES` yourself** — SLURM has already set it correctly, and
overriding it is the most common way to break your own job.

---

## 3. Wall time and queueing

Always pass a realistic time limit with `-t`. The scheduler uses backfill: an
honest short estimate lets your job slot into a gap ahead of larger jobs, so
accurate estimates mean less waiting for you *and* for everyone else.

The quality-of-service class is derived automatically from your `-t`:

| QOS | Wall limit | Priority | Notes |
|---|---|---|---|
| `short` | 6 h | highest | quick runs jump the queue |
| `normal` | 2 days | normal | job is killed at the limit |
| `long` | unlimited | lowest | max 5 GPUs in long jobs cluster-wide, 3 per user |

You normally never choose a QOS by hand. Without `-t` you get the partition
default of 4 h, which lands in `short`. Override only with an explicit
`--qos=...`. Check what your job got with `squeue` or `sacct -j <id> -o QOS`.

Priority is fair-share based: heavy recent usage lowers your priority, and
that usage decays with a 7-day half-life. Nobody can monopolise the machine
for long, and a light user's job overtakes a heavy user's job automatically.

---

## 4. Running micromagnetic simulations

`magnum.np` and `neuralmag` are wrappers that submit your python script as a
SLURM job. No job script needed:

```bash
magnum.np run.py                            # interactive: 24 GB slice, 3 h
magnum.np -b run.py                         # batch: detached, output in run-<jobid>.out
magnum.np -t 12:00:00 run.py                # 12 h wall time
magnum.np --bigmem -b -t 3-00:00:00 run.py  # whole 96 GB GPU, batch, 3 days
magnum.np -l                                # list available versions
magnum.np -v v2.0.3 run.py                  # pick a specific version
magnum.np -e ~/git/project/venv run.py      # your own virtualenv
magnum.np --dry-run run.py                  # show the srun command, run nothing
magnum.np run.py --scale 2 out.h5           # arguments after the script go to the script
```

Defaults: `--gres=gpu:1g.24gb:1 -c 8 -t 3:00:00 --mem=24G`. `--bigmem` switches
to a whole GPU and `--mem=96G`. Extra options go via `$MAGNUMNP_SRUN_ARGS`
(or `$NEURALMAG_SRUN_ARGS`); they are appended after the defaults, so
duplicates override, e.g. `MAGNUMNP_SRUN_ARGS="--mem=64G -c 16"`.

Currently installed (newest is the default):

- magnum.np: `v2.2.0`, `v2.1.0`, `v2.0.3`
- neuralmag: `v1.0.0`, `v0.9.4`

**Use `-b` for anything long.** Interactive mode blocks your terminal and the
job dies with your SSH session.

---

## 5. Running anything else

Plain SLURM works normally:

```bash
# batch job on a 24 GB slice, 3 hours
sbatch --gres=gpu:1g.24gb:1 -t 3:00:00 -c 8 --mem=32G job.sh

# whole GPU for up to 2 days
sbatch --gres=gpu:full:1 -t 2-00:00:00 -c 16 --mem=64G train.sh

# very long training run (lands in QOS long)
sbatch --gres=gpu:full:1 -t 14-00:00:00 train.sh

# interactive shell on a slice
srun --gres=gpu:1g.24gb:1 -t 4:00:00 -c 8 --mem=32G --pty bash

# benchmark that must not share the machine at all
sbatch --gres=gpu:full:1 --exclusive -t 2:00:00 bench.sh
```

CPU-only jobs are welcome — just omit `--gres`. Default is 1 core and 3 GB
RAM per core; ask for more with `-c` and `--mem`.

Exceeding `--mem` briefly does not kill your job: it spills into NVMe swap
(up to 1× your allocation) and merely runs slower during the spike.

---

## 6. Monitoring

| Command | Shows |
|---|---|
| `squeue` | the queue |
| `sinfo` | node and GPU availability |
| `sprio -l` | why jobs are ordered the way they are |
| `sshare -a` | your fair-share standing |
| `nvidia-smi` | GPU state — works for everyone, no sudo |
| `nvitop` | live GPU/process monitor |
| `sacct -j <id>` | what a finished job actually did |
| `sreport cluster UserUtilizationByAccount start=2026-08-01` | usage stats |

`nvidia-smi` and `nvitop` run read-only as an unprivileged monitoring user,
so they show everything but cannot kill anyone's processes — including your
own. Use `scancel <jobid>` for that.

---

## 7. Storage and quotas

| Location | Quota (soft/hard) | Use it for |
|---|---|---|
| `/home/<user>` | 18 / 20 GB | code, scripts, configs |
| `/data/<user>` | 180 / 200 GB | simulation output, datasets |

`/data` is the fast NVMe volume and is where results belong. Check your usage
with `quota -s`.

**Neither is backed up.** Anything you cannot regenerate belongs somewhere
else as well.

---

## 8. Login node etiquette

The login shell is the same machine as the compute node, so a heavy process
in your SSH session competes with real jobs. Login sessions are therefore
capped at roughly 22–25% of RAM per user (SLURM jobs are unaffected).

Please do not run simulations, big compiles, or data crunching directly in
your shell — submit them. Editing, plotting, and small analysis are fine.

Docker is restricted to administrators. Docker group membership is
root-equivalent and bypasses the GPU access control entirely, so GPU work
runs through SLURM instead. If you have a containerised workflow that needs a
home here, talk to an admin.
