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
  __init__.py       CLI + training loop (entrypoint: main)
  config.py         DATA_ROOT + dataset_dir(split, size) path helper
  train_config.py   TrainConfig dataclass (run hyperparameters) + sp4_path property
  data.py           trajectory loading + grain dataloader
  model.py          LLGEmulator (ResNet backbone + FiLM conditioning) + ModelConfig
  physics.py        DemagField — exact demag field via neuralmag (frozen, non-trainable)
  training.py       loss, jitted update/eval steps, train/val epoch loops
  rollout.py        autoregressive trajectory unroll (lax.scan)
  metrics.py        MSE, correlation, bulk magnetization
  plotting.py       plotly (used) + matplotlib (unused) figures for wandb
main.ipynb          scratch notebook (stale — see Issues)
scripts/run         local single-GPU run (uv run train ...)
scripts/run.slrm    SLURM batch script (sbatch scripts/run.slrm, via `make train`)
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

**Training** (`training.py`) — one-step MSE. `update_fn`/`evaluate_fn` are `filter_jit` with
donation. Every `checkpoint_every` epochs the loop logs rollout plots (bulk magnetization vs. the
SP4 reference at `train_config.sp4_path`, correlation-vs-step) to wandb. **Weights are not saved** —
`checkpoint_every` only gates the eval plots, and the artifact-saving helper was removed; a run
produces no reloadable model.

## Running

```bash
uv sync                    # install deps into .venv
uv run train --epochs 64 --batch-size 20 --learning-rate 1e-5 --size small --wandb-mode disabled
```

`--epochs`, `--batch-size`, `--learning-rate` are **required**. `--size` ∈ {small, med, large}
selects the dataset split under `DATA_ROOT`. `--hidden-channels` / `--num-blocks` set the ResNet
architecture (wired through to `ModelConfig`); everything else on the parser has a default. See
`scripts/run` for a working local invocation, or `make train` to submit `scripts/run.slrm` via SLURM.

**Data** lives in the sibling repo `../micromagnetic-data/data/dynamics/<size>/{train,val}/`
(path hard-coded in `config.py`). Only the `small` split is present on this machine
(24 train / 14 val samples; each `m.npy` is `(101, 256, 256, 1, 3)`).

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

1. **No checkpointing.** `--checkpoint-every` only gates the *eval plots*; model weights are never
   saved and a run produces no reloadable artifact. (The `wandb_io.py` save/fetch helpers were
   removed as dead + inconsistent — re-add a `save_weights` call in the loop if you need reload.)

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
