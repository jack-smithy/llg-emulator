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
  __init__.py       CLI + training loop (entrypoint: main); saves a ckpt every --checkpoint-every
  config.py         DATA_ROOT + dataset_dir(split, size) path helper
  train_config.py   TrainConfig dataclass (run hyperparameters) + sp4_path property
  data.py           trajectory loading (eager) + grain dataloader
  model.py          LLGEmulator (ResNet backbone + FiLM conditioning) + ModelConfig
  physics.py        DemagField — exact demag field via neuralmag (frozen, non-trainable)
  training.py       one-step MSE loss + jitted update, cosine sched, trainable filter
  io.py             save_model / load_model (+ .vtr writer)
  rollout.py        autoregressive trajectory unroll (lax.scan)
  metrics.py        MSE, correlation, bulk magnetization, sp4 rollout RMSE
  plotting.py       matplotlib (used by the train loop) + plotly (for wandb; currently unused)
  evaluate.py       standalone rollout-correlation eval of a saved checkpoint
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

**Data** (`data.py`) — each sample dir is `sample-{i:05d}-of-{n:05d}/` with `m.npy`
`(t, 256, 256, 3)` float32 (**nodal**, see above) + `metadata.json` (`n` = cell counts `[255,255]`,
`dx`, `dt`, `t_tot`, `material{Ms,A,alpha}`, `H_ext` in A/m, `init`, `key`). Frames are uniformly
spaced at one solver step `dt` (10 ps): 101 frames over 1 ns. `_frames_to_cf` transposes to
`(t, c, h, w)` float32; `H_ext` is normalised by `Ms`.

**Loading is eager.** `load_trajectories` reads every `m.npy` fully into host RAM with a
`ThreadPoolExecutor`, so a source costs ~79 MB per trajectory (16 train + 4 val ≈ 1.6 GB for the
`check` set; the training job peaked at ~5.5 GB RSS). `LLGStepperSource` then builds a flat
`(traj, t, stride)` index over all `(m_t, m_{t+s})` pairs for the requested `strides` and returns
`{m0, m1, H, s0}` with `s0 = log2(s)` (float32). `load_trajectory` loads one *full* trajectory for
single-trajectory eval (SP4/bulk rollout). `dataloader_factory` returns a `seed -> IterDataset`
closure (grain: shuffle indices lazily, batch, drop remainder; it stacks each dict key, so `s0`
batches to `(b,)`). If a future split doesn't fit in RAM this is the thing to make streaming.

**Training** (`training.py` + `__init__.py`) — the objective is a plain **one-step MSE**
(`loss_fn`) on `(m_t, m_{t+s})` pairs; `__init__.py` builds the source with `strides=[1]`, so
`s_enc` is constantly 0 today. `update_fn` is `filter_jit` with donation; the demag tensor leaf is
kept out of the gradient by `trainable_filter`. Optimiser: `clip_by_global_norm` + `adamw`, on a
warmup+cosine schedule (`make_schedule`) when `--cosine` is passed, otherwise a constant LR.
Val is the same one-step MSE on the val split. Every `checkpoint_every` epochs the loop rolls the
model out over the SP4 trajectory (`metrics.bulk_magnetization`), writes
`<out>/rollout_epoch_{i}.png`, and saves `<out>/epoch_{i}.eqx`; the final model goes to
`<out>/weights.eqx`. `save_model` = `eqx.tree_serialise_leaves` + a `metadata.json` sidecar
(`hidden_channels`, `num_blocks`, `cond_dim`) in the same directory; `load_model(path, demag, key,
tag)` rebuilds the skeleton from it (`cond_dim` defaults to 3 for pre-timestep sidecars).
There is **no best-checkpoint selection** — every saved epoch is kept, and the last one is
`weights.eqx`. The rollout-k loss, `--strides`/`--augment` flags and `RolloutSource` described in
earlier versions of this file no longer exist in `training.py`.

## Running

Single pipeline — `uv run train` (entrypoint `main` in `__init__.py`). Logs to wandb and writes
checkpoints + SP4 rollout plots into the `--out` **directory**.

```bash
uv sync
make train           # == sbatch scripts/run.slrm  (16 ep smoke run -> weights/test/)
scripts/run          # local run, pins CUDA_VISIBLE_DEVICES
```
Direct: `uv run train --epochs 60 --batch-size 8 --learning-rate 3e-4 --cosine
--weight-decay 1e-5 --grad-clip 1.0 --hidden-channels 64 --num-blocks 4
--checkpoint-every 5 --out weights/run1 --size large --wandb-mode disabled`. Flags:
`--cosine` warmup+cosine schedule (otherwise constant LR); `--checkpoint-every N` runs the SP4
rollout eval, saves `epoch_N.eqx` and writes `rollout_epoch_N.png` every N epochs; `--out` is a
**directory** (`save_model` writes `<out>/<tag>.eqx` + `<out>/metadata.json`) — passing a `.eqx`
*file* path makes a directory with that name, and crashes outright if a file already sits there.
`--epochs`, `--batch-size`, `--learning-rate` are required. `--size` is accepted but currently
ignored by `config.dataset_dir` (see Issues). One epoch of the `check` set (16 train / 4 val
trajectories, batch 8, hidden 64) is ~21 s on a `1g.24gb` MIG slice including the per-epoch SP4
rollout eval.

To reload + evaluate a saved model: `io.load_model(dir, demag, key, tag="weights")` (the demag must
be built for the same mesh), then `metrics.bulk_magnetization(model, sp4_path)` or
`metrics.sp4_rollout_rmse(model, source)`. `scripts/eval` / `scripts/eval.slrm` still point at the
old `results/weights/best.eqx` layout and need updating before they run.

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
- Don't launch GPU/training jobs from here — CPU tasks (import checks, data inspection) are fine.

## Known issues / cleanup

Real, and documented so they aren't mistaken for intent.

1. **No best-checkpoint selection.** The loop saves *every* `--checkpoint-every` epoch plus a
   final `weights.eqx`; nothing tracks the SP4 metric to keep the best one, and nothing prunes the
   per-epoch files (16 epochs ≈ 100 MB). The SP4 rollout is only rendered to a PNG — its RMSE is
   never computed or logged.

2. **One-step loss ⇒ rollout drift.** With the rollout-k loss gone, a 16-epoch run reaches a val
   one-step MSE of 2.8e-4 (vs 8.0e-3 for the trivial `m_{t+1} = m_t` predictor — so the step map is
   genuinely learned), but the 100-step SP4 unroll still precesses and drifts out of plane. That is
   compounding error, matching Finding 2 below; more epochs + `--cosine` and/or restoring a
   multi-step loss is the fix, not a bug in the pipeline.

3. **`main.ipynb` is stale.** It calls `LLGEmulator(key=key)` (constructor now requires
   `config=ModelConfig()`) and imports `correlation_epoch` from `llg_emulator.training` (it lives in
   `metrics`). It errors as-is. It's scratch — regenerate or delete rather than trust it.

4. **Dead code.** No callers: `metrics.nRMSE`, `metrics.correlation_epoch`,
   `plotting.plot_learning_curve` / `plot_rollout_metric`, and — now that the train loop logs the
   matplotlib figure — both plotly helpers (`plot_m_means_plotly` is only referenced from a
   commented-out line in `__init__.py`). `io.write_vtr` is also unused. Delete unless you're about
   to use them.

5. **`config.dataset_dir` ignores `size`.** It returns `DATA_ROOT / split` regardless, so
   `--size {small,med,large}` currently selects nothing; `TrainConfig.large_scale_path` points at a
   `large-scale` split that does not exist (its only caller is commented out in `__init__.py`).

6. **`ModelConfig.activation` isn't JSON-serializable.** `asdict(model_config)` (logged to wandb
   config) contains the `gelu` function object. wandb tolerates it (stores a repr), but any code
   that `json.dumps` the config will raise. Store the activation as a name string if you need clean
   serialisation.
