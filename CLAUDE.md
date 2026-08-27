# llg-emulator

Neural surrogate for **Landau–Lifshitz–Gilbert (LLG) magnetization dynamics** of a thin
film. Learns a step map `m_t -> m_{t+Δt}` on a 256×256 spin grid, conditioned on a
constant applied field `H_ext` **and the step size Δt**, then unrolls it autoregressively
to emulate full trajectories produced by a micromagnetic solver. Conditioning on Δt lets
the model take steps much larger than the 10 ps solver step (predict `m_t -> m_{t+k·dt}`
for k = 1, 2, 3, … in one call; the current loss only trains stride 1).

JAX + Equinox. GPU (CUDA 12), Python 3.13, managed with `uv`.

> **Keep this file current.** After any change to structure, the CLI, the data
> contract, or the known-issues list, update the relevant section here in the same
> change. A stale CLAUDE.md is worse than none — it gets trusted.

## Layout

```
src/llg_emulator/
  train.py          CLI + training loop (entrypoint: main); saves a ckpt every --checkpoint-every
  __init__.py       deliberately empty - keeps importing a submodule cheap (see Conventions)
  config.py         DATA_ROOT + variant_dirs(split) / sp4_path() discovery
  train_config.py   TrainConfig dataclass (run hyperparameters) + sp4_path property
  data.py           memory-mapped trajectories + grain step-pair source
  model.py          LLGEmulator (ResNet + FiLM) + ModelConfig + with_mesh retargeting
  physics.py        DemagField — exact demag via neuralmag (frozen) + cached demag_for
  training.py       one-step MSE loss + jitted update, cosine sched, trainable filter
  io.py             save_model / load_model (trainable leaves only) + self-check (+ .vtr writer)
  rollout.py        autoregressive trajectory unroll (lax.scan)
  metrics.py        MSE, correlation, streaming rollout_metrics + bulk_rmse (headline)
  plotting.py       matplotlib (used by the train loop) + plotly (for wandb; currently unused)
  evaluate.py       standalone eval: SP4 rollout + val correlation + --vtr ParaView export
  symmetry.py       D4xZ2 symmetry group (exact augmentation), demag-equivariance self-check
src/datagen/
  generate.py       training-data generation (neuralmag LLG solver) + test_generate.py smoke test
.nm_cache/          cached demag tensors (gitignored), shared by datagen + physics.py
data/check/         the dataset in use (see Data below)
main.ipynb          scratch notebook (stale — see Issues)
weights/            saved models. Legacy: best.eqx / hc64_160.eqx / best_dt.eqx (+ .json sidecars)
                    were trained on the OLD cell-based data and do NOT load against today's
                    ModelConfig. New runs write a directory: <out>/{epoch_N,weights}.eqx + metadata.json
scripts/run         local run (pins CUDA_VISIBLE_DEVICES, `uv run train`)
scripts/run.slrm    SLURM batch script (sbatch scripts/run.slrm, via `make train`)
scripts/smoke.slrm  2-epoch end-to-end check (train -> SP4 rollout -> reload -> eval)
scripts/generate.slrm  SLURM datagen job
```

## How it works

**Model** (`model.py`) — `LLGEmulator.__call__(m0, cond)`:
1. `demag(m0)` computes the long-range magnetostatic field exactly (physics, not learned).
2. `[m0, demag(m0)]` (6 channels) feeds a `pdequinox` `ClassicResNet` (Neumann boundary).
3. FiLM conditioning: the conditioning vector `cond` → per-(stage, channel) `(gamma, beta)`,
   applied after lifting and after every block. Zero-initialised final layer ⇒ identity modulation
   at start. `cond` is a single vector of length `ModelConfig.cond_dim`, built by the caller:
   `[H_ext/Ms]` (3) for the original fixed-dt model, or `[H_ext/Ms, s_enc]` (4) when `cond_dim==4`
   — `s_enc = log2(stride)` is the **step-size input** (stride in base 10 ps steps; `Δt = stride·dt`,
   `s_enc=0` ⇒ one base step). Every call site (`training.loss_fn`, `rollout_trajectory`,
   `metrics.bulk_rollout`) concatenates the field and `s_enc` itself; the model does not.
4. `step()` predicts residual `dm`, projects it into the tangent space (⊥ to `m`), adds, renormalises.
   Output is always a unit-norm field. (Geometric, so it handles the larger `|dm|` of a big jump.)

**Physics** (`physics.py`) — `DemagField` precomputes neuralmag's demag tensor `N` once at
construction (mesh geometry only) and applies an FFT convolution per call. Pure Equinox module,
no learnable params; the tensor leaf is **frozen** during training (`trainable_filter` in
`training.py`). Output is nondimensionalised by `Ms`, matching the `H_ext / Ms` input convention.
`Ms` cancels under this nondim, so the model builds demag with `Ms=1.0`. The tensor is cached on
disk in `.nm_cache/` (neuralmag's `DemagField(cache_dir=...)`, keyed on n/dx/pbc/p), so repeated
model construction skips the ~O(minutes) build. neuralmag assembles the tensor in float64 internally
regardless of `nm.config.dtype` and casts on load, so float32 elsewhere costs no kernel accuracy.

**Nodal vs. cell discretization (easy to get wrong).** The solver runs on a **2D mesh of
`n = (nx, ny)` cells** and stores `m` on the **nodes** — `(nx+1, ny+1)` values, which is what the
`.npy` files contain (`n = (255, 255)` cells ⇒ 256×256 nodal grid). neuralmag's convolution is
cell-based, so `DemagField.__call__` follows neuralmag's own node kernel:
`m_node --to_cell--> m_cell --h_cell--> h_cell --to_node_w--> h_node`. Both mass-lumped projections
are generated by neuralmag for this mesh and resolved once at construction (stored as static
fields). Verified against `state.h_demag` to ~1e-7 relative, jitted and vmapped. **Skipping the
projections** (treating the 256×256 nodal grid as a 256×256 cell grid, which is what the code did
when the data was cell-based) is not a small approximation — it gets the field wrong by >100%.
`DemagField(n, dx, …)` therefore takes the **cell** count `(nx, ny)` and returns `(3, nx+1, ny+1)`.

**Data** (`data.py`) — the layout is `data/<split>/<variant>/sample-{i:05d}-of-{n:05d}/`
with `m.npy` + `metadata.json`. **Every subdirectory of a split is a variant** and is picked up
automatically — `config.variant_dirs(split)` enumerates them, so adding a variant needs no code
change. Current contents:

| split / variant | trajectories | mesh `n` (cells) | `m.npy` shape (nodal) |
|---|---|---|---|
| `train/fixed_geo` | 512 | (256, 256) | (101, 257, 257, 3) |
| `val/fixed_geo`   | 256 | (256, 256) | (101, 257, 257, 3) |
| `val/sp4`         | 1   | **(100, 25)** | (101, 101, 26, 3) |
| `val/large`       | 1   | **(2000, 2000)** | (101, 2001, 2001, 3) |

`metadata.json` carries `n` (cell counts), `dx`, `dt`, `t_tot`, `material{Ms,A,alpha}`, `H_ext`
in A/m, `init`, `key`. Frames are uniformly spaced at one solver step `dt` (10 ps): 101 frames
over 1 ns. `H_ext` is normalised by `Ms`; `_frames_to_cf` transposes a whole trajectory to
`(t, c, h, w)` float32, `_frame_to_cf` does one frame.

**The benchmark variants (`config.BENCHMARK_VARIANTS` = `sp4`, `large`) are held out of the val
loss.** That is not just convention — each sits on its own mesh (100x25 and 2000x2000 cells against
the training set's 256x256), so neither *can* share a batch with the other val trajectories. `sp4`
tests field extrapolation plus a reversal; `large` is 61x the training area and tests domain-size
transfer alone. Adding a variant to `data/val/` that is *not* listed in `BENCHMARK_VARIANTS` and is
not on the training mesh will make training fail at startup on the mixed-mesh guard.

**Loading is memory-mapped.** The dataset is 59 GB over 769 trajectories; one training sample
touches two frames (~0.8 MB). `open_trajectory` mmaps `m.npy` and reads the metadata without
touching the frames (mapping all 769 costs ~30 MB RSS), and `LLGStepperSource.__getitem__`
transposes the two frames it needs on demand — the OS page cache does the caching. `load_trajectory`
still reads one *full* trajectory eagerly, for single-trajectory eval where every frame is needed.

`LLGStepperSource` builds a flat `(traj, t, stride)` index over all `(m_t, m_{t+s})` pairs for the
requested `strides` and returns `{m0, m1, H, s0}`. **`s0` is in physical units**:
`s0 = log2(stride * dt / BASE_STEP_TIME)`, so `s0 == 0` is one 10 ps step whatever `dt` a future
variant is generated at. The source **requires one mesh across all its trajectories** (grain stacks
samples into a batch; a batch of mixed grids has no shape) and raises naming the fix if a split
mixes them. `dataloader_factory(source, batch_size)` returns a `seed -> IterDataset` closure
(grain: shuffle lazily, batch, drop remainder; it stacks each dict key, so `s0` batches to `(b,)`).

**Mesh invariance.** The backbone is fully convolutional, so it transfers to any grid unchanged;
the demag tensor is the *only* mesh-shaped part of the model. `model.with_mesh(model, n, dx)`
returns the same weights with a demag rebuilt for `(n, dx)`, and refuses a different `dx` — a
different cell size changes the physics per cell, which no rebuild fixes. `physics.demag_for` is
`lru_cache`d, which matters twice: the build is O(minutes) on an `.nm_cache` miss, and the generated
projection kernels are *static* pytree fields, so a fresh `DemagField` per call would retrigger jit
compilation on every rollout. Eval paths run one trajectory at a time and retarget per trajectory,
so **mixed-mesh evaluation already works**; mixed-mesh *training* would need one batch stream and
one demag per mesh, and is not implemented.

**Training** (`training.py` + `train.py`) — the objective is a plain **one-step MSE**
(`loss_fn`) on `(m_t, m_{t+s})` pairs; `train.py` builds the source with `strides=[1]`, so
`s_enc` is constantly 0 today. Val uses `with_mesh`, so a val split on another mesh still scores.
`update_fn` is `filter_jit` with donation; the demag tensor leaf is
kept out of the gradient by `trainable_filter`. Optimiser: `clip_by_global_norm` + `adamw`, on a
warmup+cosine schedule (`make_schedule`) when `--cosine` is passed, otherwise a constant LR.
Val is the same one-step MSE on the val split. Every `checkpoint_every` epochs the loop rolls the
model out over the SP4 trajectory (`metrics.bulk_magnetization`), writes
`<out>/rollout_epoch_{i}.png`, and saves `<out>/epoch_{i}.eqx`; the final model goes to
`<out>/weights.eqx`. `save_model` serialises **only the trainable leaves** (`eqx.partition` on
`trainable_filter`) plus a `metadata.json` sidecar (`hidden_channels`, `num_blocks`, `cond_dim`)
in the same directory; `load_model(path, demag, key, tag)` rebuilds the skeleton from the sidecar,
deserialises into the filtered tree and `eqx.combine`s the caller's `demag` back in (`cond_dim`
defaults to 3 for pre-timestep sidecars). The frozen demag tensor is **not** written — it was 78%
of a 255x255 checkpoint (6.06 MB -> 1.36 MB) and is rebuilt exactly from the mesh on load, so
checkpoints saved before this change (full-tree) no longer load. `uv run python -m llg_emulator.io`
round-trips a small model as a self-check.
Every `checkpoint_every` epochs the SP4 bulk RMSE is computed and logged as `sp4/bulk_rmse`, and
the best-so-far model is saved as `<out>/best.eqx` (`wandb.summary.best_sp4_bulk_rmse` /
`best_epoch`). Every checkpointed epoch is *also* kept, and the last one is `weights.eqx`. The rollout-k loss, `--strides`/`--augment` flags and `RolloutSource` described in
earlier versions of this file no longer exist in `training.py`.

## Running

Single pipeline — `uv run train` (entrypoint `main` in `train.py`). Logs to wandb and writes
checkpoints + SP4 rollout plots into the `--out` **directory**.

```bash
uv sync
make train           # == sbatch scripts/run.slrm  (16 ep smoke run -> weights/test/)
scripts/run          # local run, pins CUDA_VISIBLE_DEVICES
```
Direct: `uv run train --epochs 40 --batch-size 8 --learning-rate 3e-4 --cosine
--weight-decay 1e-5 --grad-clip 1.0 --hidden-channels 64 --num-blocks 4
--checkpoint-every 5 --out weights/run1 --wandb-mode disabled`. Flags: `--cosine` warmup+cosine
schedule (otherwise constant LR); `--checkpoint-every N` runs the SP4 rollout eval, logs
`sp4/bulk_rmse`, saves `epoch_N.eqx` (plus `best.eqx` on improvement) and writes
`rollout_epoch_N.png`; `--max-train-trajectories` / `--max-val-trajectories` cap how many
trajectories each split maps, for smoke runs; `--out` is a **directory** (`save_model` writes
`<out>/<tag>.eqx` + `<out>/metadata.json`) — passing a `.eqx` *file* path makes a directory with
that name, and crashes outright if a file already sits there. `--epochs`, `--batch-size`,
`--learning-rate` are required. **The mesh comes from the training data**, not from a flag.

`sbatch scripts/smoke.slrm` is the end-to-end check: 2 epochs on 8 train / 4 val trajectories, the
SP4 cross-mesh rollout, a `best.eqx` reload and an eval pass. ~40 s on a `1g.24gb` MIG slice; output
in `logs/smoke.log`.

To reload + evaluate a saved model: `io.load_model(dir, key, tag="best")` — the checkpoint is
self-describing (the sidecar carries `mesh_n`/`mesh_dx`/`demag_p`, so the demag is rebuilt for you;
pass `n=` to retarget another mesh). Then `metrics.bulk_magnetization(model, sp4_path())` and
`metrics.bulk_rmse`. `sbatch scripts/eval.slrm`, or `scripts/eval <dir> <tag>`, runs
`llg_emulator.evaluate`: SP4 bulk RMSE + plot, plus rollout correlation over val trajectories.
Benchmarks run through `metrics.rollout_metrics`, which streams the rollout one frame at a time
(bulk ⟨m⟩(t) plus per-frame correlation) — `val/large` is 4.85 GB, so its full history does not fit
on the device. `--vtr` stays SP4-only for the same reason: the three `large` series would be 14.5 GB.
`--vtr` additionally writes `<out>/vtr/sp4_{reference,predicted,error}.pvd` — one `.vtr` per
frame plus a PVD collection carrying the physical time, so ParaView scrubs the 1 ns directly.
`m` is nodal and `io.write_vtr` writes cell data, so the nodal counts are passed as the cell
counts: each node renders as one voxel, offset half a cell. Nothing is interpolated.

### Current results

`BENCHMARKS.md` has the numbers. Headline: **`run1`, sp4_bulk_rmse 0.0177** (hidden 64,
4 blocks, 40 epochs, one-step loss, cosine, lr 3e-4) — trained on 256x256, evaluated on SP4's
100x25 mesh, val one-step MSE 2.26e-06 against an 8.0e-3 trivial baseline. Nothing has overfit
at 40 epochs on 512 trajectories, so the capacity and epoch ceilings below no longer apply.

### Best recipe & results (SP4) — historical

> ⚠️ These numbers are from the **old dataset** (cell-based `m` on the 3D `(256,256,1)` mesh, in
> `../micromagnetic-data`) and the **rollout-k loss that is no longer in `training.py`**. They are
> kept as the record of what mattered; they are not reproducible with the current code/data as-is,
> and the checkpoints in `weights/*.eqx` do not load against today's `ModelConfig`.

The numbers below are the **fixed-dt** sweep (stride-1 only); timestep conditioning is a separate,
newer axis (its own eval is `sp4_bulk_rmse@{5,10}` — big-step rollout). Sweep results are in
`BENCHMARKS.md`. Headline metric: `sp4_bulk_rmse`, the RMSE of the bulk magnetization ⟨m⟩(t) over a
100-step autoregressive rollout vs. the SP4 reference (whose applied field is **out-of-distribution**
— more negative Hₓ than any training sample). The SP4 switching frame (6) is reproduced exactly by
all decent models.

- **Winner** (`weights/best.eqx`): hidden 64, num_blocks 4, **rollout-k4 loss**, cosine schedule,
  lr 3e-4, wd 1e-5, grad-clip 1, best-checkpointed on SP4 rollout. **sp4_bulk_rmse 0.037**,
  val_corr 0.982 (best rollout stability). Reference config was 0.81 → **~22× better**.
- `weights/hc64_160.eqx`: same but one-step / 160 ep — marginally lower bulk RMSE (0.034) but
  worse late-time error and rollout stability. Kept as the accuracy-optimal alternative.

**Findings** (what moved SP4, from the sweep):
1. **Training budget dominated.** The reference config (lr 1e-5, 8 ep) was badly undertrained;
   cosine + lr 3e-4 + more epochs alone took 0.81 → 0.08.
2. **Rollout-k4 loss** (train on a 4-step unroll, backprop through it) fixed the post-switch
   precession and gave the most stable rollouts. k=2 was too short and *hurt* (0.14).
3. **Capacity helps to a point:** hidden 32→64 improves; **hidden 128 overfits** the 24
   trajectories (erratic, 0.11). Likewise **>160 epochs overfits** (240 ep = 0.046 > 160 ep 0.034).
4. **D4×Z₂ symmetry augmentation** is physically exact (`symmetry.py` verifies demag-equivariance)
   and 16× — but didn't beat plain training: a well-trained model already extrapolates to SP4's
   field. Left in as a flag, not in the winning recipe.
5. Held-out val correlation tracks sp4_bulk_rmse at **corr −0.97** across all runs → the SP4 gains
   are genuine generalisation, not tuning to the SP4 selection metric.

**Data** lives in `data/check/{train,val,sp4}/` (`DATA_ROOT` in `config.py`; `dataset_dir`
currently ignores its `size` argument). 16 train / 4 val trajectories + 1 SP4 reference, each
`m.npy` `(101, 256, 256, 3)` float32 nodal. SP4 reference is
`data/check/sp4/sample-00001-of-00001` (`train_config.sp4_path`). Regenerate with
`src/datagen/generate.py` (`sbatch scripts/generate.slrm`).

## Conventions

- **JAX/Equinox**: modules are pytrees; use `eqx.filter_jit` + `eqx.partition`/`combine` to keep
  the frozen demag tensor out of grads. Models are written for a single sample and `vmap`-ed for
  batches (see `loss_fn`, `rollout_trajectories`).
- **Shapes**: fields are channel-first `(3, nx, ny)` per sample, `(b, 3, nx, ny)` batched,
  `(t, 3, nx, ny)` for trajectories — where `(nx, ny)` is the **nodal** grid, i.e. `mesh_n + 1`
  in each direction. `H_ext` is a plain `(3,)` vector; the FiLM input is `cond = [H_ext/Ms, s_enc]`.
- **Nondimensionalisation**: all fields are in units of `Ms` (both `H_ext` input and demag output).
- **Keep `__init__.py` empty.** Anything at package-root scope runs on *every* submodule import:
  when the training loop lived there, `from llg_emulator.config import DATA_ROOT` cost 1.2 s and
  pulled in wandb, matplotlib, plotly, optax and neuralmag. Entrypoints live in their own modules
  (`train.py`, `evaluate.py`), which is also where the `jax.config.update` side effects belong.
- Don't launch GPU/training jobs from here — CPU tasks (import checks, data inspection) are fine.

## Known issues / cleanup

Real, and documented so they aren't mistaken for intent.

1. **Checkpoints are never pruned.** The loop saves *every* `--checkpoint-every` epoch plus
   `best.eqx` and `weights.eqx` (1.36 MB each at 256x256, hidden 64). Best-checkpoint selection on
   `sp4/bulk_rmse` now exists; pruning the per-epoch files does not.

2. **~~One-step loss ⇒ rollout drift.~~ Largely resolved by data + budget.** The 16-epoch run this
   issue described drifted out of plane over the 100-step unroll. `run1` (40 epochs, 512
   trajectories, cosine) reaches **sp4_bulk_rmse 0.0177** with the plain one-step loss and tracks the
   reference through the switch and the full precession tail — see `BENCHMARKS.md`. The rollout-k
   loss is worth re-testing but is no longer needed for a stable unroll. Note `sp4_bulk_rmse` is
   *not* monotonic in the val loss (0.053 -> 0.123 between epochs 10 and 20 while both losses fall),
   which is why best-checkpointing on it matters.

3. **Mixed-mesh training is not supported.** `LLGStepperSource` raises if one split mixes meshes,
   because a batch of mixed grids has no shape. Evaluation is already mesh-agnostic (one trajectory
   at a time + `with_mesh`). When a variant on a second mesh arrives, the fix is one batch stream
   and one `demag_for` per mesh, interleaved — not a change to the model.

4. **`main.ipynb` is stale.** It loads `data/256x256x1/…`, a path that no longer exists, and calls
   `LLGEmulator(key=key)` (the constructor requires `config=ModelConfig()`). It errors as-is. It's
   scratch — regenerate or delete rather than trust it.

5. **Dead code.** No callers: `metrics.nRMSE`, `metrics.MSE`, `rollout.rollout_trajectories`,
   `plotting.plot_corr_plotly` / `plot_learning_curve` / `plot_rollout_metric`, and
   the whole of `symmetry.py` (the `--augment` flag it served is gone; its `_demo` demag-equivariance
   self-check is the part worth keeping). `plot_m_means_plotly` and `TrainConfig.large_scale_path`
   are referenced only from commented-out blocks in `train.py`. Delete unless you're about to use
   them.

6. **`README.md` is empty** and `pyproject.toml` still carries the `uv init` placeholder
   description. `.ruff_cache/` is in the tree but ruff is not a dependency.
