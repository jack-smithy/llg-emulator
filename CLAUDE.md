# llg-emulator

Neural surrogate for **Landau–Lifshitz–Gilbert (LLG) magnetization dynamics** of a thin
film. Learns a one-step map `m_t -> m_{t+1}` on a 256×256 spin grid, conditioned on a
constant applied field `H_ext`, then unrolls it autoregressively to emulate full
trajectories produced by a micromagnetic solver.

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
main.ipynb          scratch notebook (stale — see Issues)
weights/            saved models: best.eqx (winner) + hc64_160.eqx, each with a .json sidecar
scripts/run         local run: auto-picks lowest-mem GPU, trains winning recipe (uv run train)
scripts/run.slrm    SLURM batch script (sbatch scripts/run.slrm, via `make train`)
BENCHMARKS.md       readable results table (one row per sweep run); winner marked
```

## How it works

**Model** (`model.py`) — `LLGEmulator.__call__(m0, H)`:
1. `demag(m0)` computes the long-range magnetostatic field exactly (physics, not learned).
2. `[m0, demag(m0)]` (6 channels) feeds a `pdequinox` `ClassicResNet` (Neumann boundary).
3. FiLM conditioning: `H_ext` (3-vec) → per-(stage, channel) `(gamma, beta)`, applied after
   lifting and after every block. Zero-initialised final layer ⇒ identity modulation at start.
4. `step()` predicts residual `dm`, projects it into the tangent space (⊥ to `m`), adds, renormalises.
   Output is always a unit-norm field.

**Physics** (`physics.py`) — `DemagField` precomputes neuralmag's demag tensor `N` once at
construction (mesh geometry only) and applies an FFT convolution per call. Pure Equinox module,
no learnable params; the tensor leaf is **frozen** during training (`trainable_filter` in
`training.py`). Output is nondimensionalised by `Ms`, matching the `H_ext / Ms` input convention.
`Ms` cancels under this nondim, so the model builds demag with `Ms=1.0`.

**Data** (`data.py`) — each sample dir has `m.npy` `(t, h, w, 1, 3)` + `params.json`. Loading
squeezes `nz=1`, transposes to `(t, c, h, w)` float32, and normalises `H_ext` by `Ms`.
`LLGStepperSource` flattens all trajectories into consecutive `(m_t, m_{t+1}, H)` pairs for
one-step training. `dataloader_factory` returns a `seed -> IterDataset` closure (grain: shuffle,
batch, host→device prefetch).

**Training** (`training.py` + `__init__.py`) — the objective is a **k-step rollout MSE**
(`rollout_loss_fn`): the model is unrolled on its own predictions for `k` steps and every
intermediate frame is matched, so the gradient sees the compounding error one-step training
ignores (k=1 recovers plain one-step MSE). `update_fn` is `filter_jit` with donation; the
optimiser is `clip_by_global_norm` + `adamw` on a warmup+cosine schedule (`make_schedule`).
Train data comes from `RolloutSource` (length-(k+1) windows, optional D4×Z₂ augmentation); val is
one-step MSE on `LLGStepperSource`. Every `checkpoint_every` epochs the loop logs rollout plots +
correlation to wandb, computes `sp4_rollout_rmse`, and **saves the best-SP4 checkpoint** to
`--out` (via `save_model`: `eqx.tree_serialise_leaves` + a `.json` arch sidecar; reload with
`load_model`).

## Running

Single pipeline — `uv run train` (entrypoint `main` in `__init__.py`). Trains the winning recipe,
logs to wandb, and saves the best-SP4-rollout checkpoint to `--out`.

```bash
uv sync
scripts/run          # auto-picks lowest-mem GPU, trains winner -> weights/best.eqx
```
Direct: `uv run train --epochs 60 --batch-size 8 --learning-rate 3e-4 --rollout-k 4 --cosine
--weight-decay 1e-5 --grad-clip 1.0 --hidden-channels 64 --checkpoint-every 5 --out weights/best.eqx
--wandb-mode disabled`. Flags: `--rollout-k K` trains on a K-step unroll (K=1 = one-step MSE);
`--cosine` warmup+cosine schedule; `--augment` D4×Z₂ symmetry aug (16×, CPU-bound, didn't help —
see Findings); `--checkpoint-every N` runs the SP4 rollout eval every N epochs and **keeps the
best-rollout checkpoint** (not the last). Recipe defaults (hidden 64, rollout-k 4, wd/clip) are the
sweep winner; `--epochs`, `--batch-size`, `--learning-rate` are required. `--size` ∈ {small, med,
large}. Always run on the lowest-memory GPU via `CUDA_VISIBLE_DEVICES` (`scripts/run` does this).

To reload + evaluate a saved model: `training.load_model(path)` then
`training.sp4_rollout_rmse(model, sp4_path)`.

### Best recipe & results (SP4)

Sweep results are in `BENCHMARKS.md`. Headline metric: `sp4_bulk_rmse`, the RMSE of the bulk
magnetization ⟨m⟩(t) over a 100-step autoregressive rollout vs. the SP4 reference (whose applied
field is **out-of-distribution** — more negative Hₓ than any training sample). The SP4 switching
frame (6) is reproduced exactly by all decent models.

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
