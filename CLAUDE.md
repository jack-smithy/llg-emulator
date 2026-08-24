# llg-emulator

Neural surrogate for **Landau–Lifshitz–Gilbert (LLG) magnetization dynamics** of a thin
film. Learns a step map `m_t -> m_{t+Δt}` on a 256×256 spin grid, conditioned on a
constant applied field `H_ext` **and the step size Δt**, then unrolls it autoregressively
to emulate full trajectories produced by a micromagnetic solver. Conditioning on Δt lets
the model take steps much larger than the 10 ps solver step (predict `m_t -> m_{t+k·dt}`
for k = 1, 2, 3, … in one call).

JAX + Equinox. GPU (CUDA 12), Python 3.13, managed with `uv`.

> **Keep this file current.** After any change to structure, the CLI, the data
> contract, or the known-issues list, update the relevant section here in the same
> change. A stale CLAUDE.md is worse than none — it gets trusted.

## Layout

```
src/llg_emulator/
  __init__.py       CLI + training loop (entrypoint: main) — winning recipe, saves best-SP4 ckpt
  config.py         DATA_ROOT + dataset_dir(split, size) path helper
  train_config.py   TrainConfig dataclass (run hyperparameters) + sp4_path property
  data.py           trajectory loading + grain dataloader
  model.py          LLGEmulator (ResNet backbone + FiLM conditioning) + ModelConfig
  physics.py        DemagField — exact demag field via neuralmag (frozen, non-trainable)
  training.py       rollout-k loss + jitted update, RolloutSource, cosine sched, save/load, SP4 ckpt metric
  rollout.py        autoregressive trajectory unroll (lax.scan)
  metrics.py        MSE, correlation, bulk magnetization
  plotting.py       plotly (used) + matplotlib (unused) figures for wandb
  symmetry.py       D4xZ2 symmetry group (exact augmentation), demag-equivariance self-check
src/datagen/
  generate.py       training-data generation (neuralmag LLG solver) + test_generate.py smoke test
.nm_cache/          cached demag tensors (gitignored), shared by datagen + physics.py
main.ipynb          scratch notebook (stale — see Issues)
weights/            saved models: best.eqx (fixed-dt winner) + hc64_160.eqx; best_dt.eqx
                    (timestep-conditioned, from scripts/run). Each has a .json sidecar (arch + cond_dim)
scripts/run         local run: auto-picks lowest-mem GPU, trains winning recipe (uv run train)
scripts/run.slrm    SLURM batch script (sbatch scripts/run.slrm, via `make train`)
BENCHMARKS.md       readable results table (one row per sweep run); winner marked
```

## How it works

**Model** (`model.py`) — `LLGEmulator.__call__(m0, H)`:
1. `demag(m0)` computes the long-range magnetostatic field exactly (physics, not learned).
2. `[m0, demag(m0)]` (6 channels) feeds a `pdequinox` `ClassicResNet` (Neumann boundary).
3. FiLM conditioning: the conditioning vector → per-(stage, channel) `(gamma, beta)`, applied after
   lifting and after every block. Zero-initialised final layer ⇒ identity modulation at start.
   The vector is `H_ext` (3) when `ModelConfig.cond_dim==3` (original fixed-dt model), or
   `[H_ext, s_enc]` (4) when `cond_dim==4` — `s_enc = log2(stride)` is the **step-size input**
   (stride in base 10 ps steps; `Δt = stride·dt`). `__call__(m0, H, s_enc=0)` defaults to stride 1.
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

**Data** (`data.py`) — each sample dir has `m.npy` `(t, h, w, 1, 3)` + `params.json`. Frames are
uniformly spaced at one solver step `dt` (10 ps): 101 frames over 1 ns. `_frames_to_cf` squeezes
`nz=1`, transposes to `(t, c, h, w)` float32; `H_ext` is normalised by `Ms`.

**Streaming** — the training sources do **not** load `m.npy` into RAM. `TrajectoryStore` reads only
each `params.json` (H, dt) and the `m.npy` *header* (frame count) at construction, then
**memory-maps** files and slices just the frames a sample needs on `__getitem__` (handles cached per
file; the grain pipeline is thread-based, so a shared cache is safe). Host memory stays O(index + a
few live frames) — constructing a source is ~1 MB regardless of dataset size, vs ~79 MB/trajectory
if fully loaded. This is what lets training scale past RAM (the old eager loader OOM'd on the large
dataset). `LLGStepperSource` streams consecutive `(m_t, m_{t+1}, H, s_enc=0)` one-step pairs (val);
`RolloutSource` streams strided rollout windows (train). `load_trajectory` still eagerly loads one
*full* trajectory — used only for single-trajectory eval (SP4/bulk rollout), where that's correct.
`correlation_epoch` streams one val trajectory at a time (no stacking the whole set).
`dataloader_factory` returns a `seed -> IterDataset` closure (grain: shuffle indices lazily, batch,
bounded host→device prefetch via `cpu_buffer_size`/`device_buffer_size`; it stacks each dict key, so
`s_enc` batches to `(b,)`).

**Training** (`training.py` + `__init__.py`) — two nested notions of step (don't conflate):
- **stride `s`** — the physical jump `Δt = s·dt`; the model predicts `m_t -> m_{t+s}` in one call
  and is conditioned on `s_enc = log2(s)`. `--strides` is the set trained on (default `1 2 4 8`).
- **rollout-K** (`--rollout-k`, default 4) — how many autoregressive steps the *loss* unrolls.

Objective: a **K-step rollout MSE over strided windows** (`rollout_loss_fn` + `RolloutSource`,
window `trj[t : t+K·s+1 : s]`), so the unroll is over the big-step map — every intermediate frame
matched, gradient sees compounding error (K=1 = single big-step MSE). `update_fn` is `filter_jit`
with donation; optimiser is `clip_by_global_norm` + `adamw` on a warmup+cosine schedule
(`make_schedule`); optional D4×Z₂ augmentation. Val is one-step MSE on `LLGStepperSource`. Every
`checkpoint_every` epochs the loop logs rollout plots + correlation, computes
`sp4_rollout_rmse(stride=1)` (the checkpoint metric) plus big-step `sp4_bulk_rmse@{5,10}`, and
**saves the best-SP4 checkpoint** to `--out` (`save_model`: `eqx.tree_serialise_leaves` + a `.json`
arch sidecar storing `cond_dim`; reload with `load_model`, which defaults `cond_dim=3` for
pre-timestep checkpoints).

## Running

Single pipeline — `uv run train` (entrypoint `main` in `__init__.py`). Trains the winning recipe
(now timestep-conditioned), logs to wandb, saves the best-SP4-rollout checkpoint to `--out`.

```bash
uv sync
scripts/run          # auto-picks lowest-mem GPU, trains dt-model -> weights/best_dt.eqx
```
Direct: `uv run train --epochs 60 --batch-size 8 --learning-rate 3e-4 --rollout-k 4
--strides 1 2 4 8 --cosine --weight-decay 1e-5 --grad-clip 1.0 --hidden-channels 64
--checkpoint-every 5 --out weights/best_dt.eqx --wandb-mode disabled`. Flags:
`--strides S…` step sizes trained on (predict `m_t -> m_{t+s·dt}`, conditioned on `s`; use
`--strides 1` for the original fixed-dt map); `--rollout-k K` unroll length in the loss (K=1 =
single big-step MSE); `--cosine` warmup+cosine schedule; `--augment` D4×Z₂ symmetry aug (16×,
CPU-bound, didn't help — see Findings); `--checkpoint-every N` runs the SP4 rollout eval every N
epochs and **keeps the best-rollout checkpoint** (not the last). Recipe defaults (hidden 64,
rollout-k 4, wd/clip) are the sweep winner; `--epochs`, `--batch-size`, `--learning-rate` are
required. `--size` ∈ {small, med, large}. Always run on the lowest-memory GPU via
`CUDA_VISIBLE_DEVICES` (`scripts/run` does this).

To reload + evaluate a saved model: `training.load_model(path)`, then
`training.sp4_rollout_rmse(model, sp4_path, stride=s)` — `stride=1` is the 10 ps rollout;
larger `stride` takes big `s·10` ps steps (the point of timestep conditioning). `load_model`
reads `cond_dim` from the sidecar, so the pre-timestep `weights/best.eqx` (no `cond_dim`) still
loads as the fixed-dt baseline (0.037).

### Best recipe & results (SP4)

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

**Data** lives in the sibling repo `../micromagnetic-data/data/dynamics/<size>/{train,val}/`
(path hard-coded in `config.py`). Only the `small` split is present on this machine
(24 train / 14 val samples; each `m.npy` is `(101, 256, 256, 1, 3)`). SP4 reference trajectory is
at `.../small/sp4/sample_0` (`train_config.sp4_path`).

## Conventions

- **JAX/Equinox**: modules are pytrees; use `eqx.filter_jit` + `eqx.partition`/`combine` to keep
  the frozen demag tensor out of grads. Models are written for a single sample and `vmap`-ed for
  batches (see `loss_fn`, `rollout_trajectories`).
- **Shapes**: fields are channel-first `(3, nx, ny)` per sample, `(b, 3, nx, ny)` batched,
  `(t, 3, nx, ny)` for trajectories. `H_ext` is a plain `(3,)` vector.
- **Nondimensionalisation**: all fields are in units of `Ms` (both `H_ext` input and demag output).
- Don't launch GPU/training jobs from here — CPU tasks (import checks, data inspection) are fine.

## Known issues / cleanup

Real, and documented so they aren't mistaken for intent.

1. *(resolved)* Checkpointing now works: the `uv run train` loop best-checkpoints on SP4 rollout
   via `training.save_model` and writes `--out` (+ a `.json` arch sidecar). `--checkpoint-every`
   gates both the eval plots and the checkpoint save.

2. **`main.ipynb` is stale.** It calls `LLGEmulator(key=key)` (constructor now requires
   `config=ModelConfig()`) and imports `correlation_epoch` from `llg_emulator.training` (it lives in
   `metrics`). It errors as-is. It's scratch — regenerate or delete rather than trust it.

3. **Dead code.** No callers: `metrics.nRMSE`, `plotting.plot_m_means` / `plot_learning_curve` /
   `plot_rollout_metric` (the plotly variants are the live ones), `DemagField.from_params`. Delete
   unless you're about to use them.

4. **`ModelConfig.activation` isn't JSON-serializable.** `asdict(model_config)` (logged to wandb
   config) contains the `gelu` function object. wandb tolerates it (stores a repr), but any code
   that `json.dumps` the config will raise. Store the activation as a name string if you need clean
   serialisation.
