# llg-emulator

Learn LLG (Landau–Lifshitz–Gilbert) magnetization dynamics for a permalloy-ish
thin film. The model is a neural surrogate that maps one magnetization field +
applied field to the next timestep, then is autoregressively rolled out to
reproduce a full trajectory. Built on JAX / Equinox using
[PDEquinox](https://github.com/Ceyron/pdequinox) for prebuilt architectures.

After each major change, update this file.

## Stack

- JAX (`jax[cuda13]`), Equinox, Optax, jaxtyping, einops, matplotlib, tqdm.
- PyTorch is used **only** for its `DataLoader`/`Dataset` plumbing; batches are
  collated into JAX arrays (`llg_emulator.data.jax_collate`).
- Python >= 3.13, managed with `uv`. `src/` layout package (hatchling); run
  `uv sync` once, then scripts via `uv run scripts/train.py` etc.
- Single-GPU: each script sets `CUDA_VISIBLE_DEVICES` (from
  `config.DEVICE`, default `3`) before importing JAX, then calls
  `jax_setup.configure_jax()` to persist the JIT cache to `.jax_cache/`.

## Data

External, not in this repo. Lives in a sibling repo at
`../micromagnetic-data/data/dynamics/v2/<size>/{train,val}` (size `small` or
`med`; `config.DATA_ROOT`/`config.dataset_dir`), plus a `sp4/sample_0`
trajectory (`config.SP4_PATH`) used for rollout eval/animation. One directory
per sample (`sample_*`), each containing:
- `m.npy` — magnetization array, shape `(T, W, H, 1, 3)`. The singleton axis
  (a z-slice) is dropped on load; trajectories are 100 timesteps.
- `params.json` — includes `H_ext` (3-vector applied field) and
  `material.Ms` (saturation magnetization). `H_ext` is nondimensionalized by
  dividing by `Ms`.

Time-axis convention (unified): frame 0 is an identical duplicate of frame 1
(the initial condition) and is **dropped everywhere**. Both
`data.LLGDataset` (zero-copy `np.load(...)[1:]` view) and
`data.load_trajectory` (`m[1:, ...]`) expose index 0 = raw frame 1, so
rollout output and references are directly comparable.

## Layout

`src/llg_emulator/` — importable library:

- `config.py` — constants only, **no jax import** (so it loads before
  `CUDA_VISIBLE_DEVICES` is set): `DEVICE`, `JAX_CACHE_DIR`, `DATA_ROOT`,
  `SIZE`, `SP4_PATH`, `RESULTS_DIR`, `dataset_dir(split, size)`.
- `jax_setup.py` — `configure_jax()` (JIT cache dir).
- `model.py` — `LLGEmulator` subclasses `pdequinox.arch.ClassicFNO` (2D FNO,
  3 in / 3 out, default 32 hidden ch, 12 modes, 4 blocks, GELU). The
  6-channel feature `[m_t (3), H_ext broadcast (3)]` is split inside the
  model: only `m_t` flows through the FNO; the constant `H_ext` vector
  drives a `_FiLM` conditioner that modulates the hidden features after the
  lifting layer and after every block (zero-init final layer ⇒ identity
  modulation at start). Residual formulation: the FNO output `dm` is
  projected onto `m_t`'s tangent plane (LLG keeps `|m|=1`), then
  `m_{t+1} = normalize(m_t + dm_perp)`. Note: weights from runs trained
  before this arch change (6-channel FNO) are no longer reloadable.
  `spherical_to_cartesian` defined but unused (kept for possible future use).
- `data.py` — `LLGDataset`, `JaxLoader`, `jax_collate`, `to_device`,
  `load_metadata`, `load_trajectory`. `_nondim_field` (H_ext / Ms) is shared
  by the torch (`LLGDataset`) and jax (`load_trajectory`) paths.
- `rollout.py` — `stepper_fn`, `rollout` (`jax.lax.scan` unroll), and
  `rollout_trajectory(model, m_true, H_ext)` (the one-call helper used by all
  three scripts).
- `metrics.py` — `nRMSE`, `correlation`.
- `training.py` — `count_parameters`, `loss_fn` (one-step MSE, single home),
  `update_fn`, `train_epoch`, `val_epoch`.
- `experiment.py` — `TrainConfig` (+ nested `ModelConfig`/`DataConfig`/
  `OptimConfig`) typed dataclasses. `from_toml` (stdlib `tomllib`, unknown
  keys raise), `save` (copies source `.toml` + writes `resolved_config.json`),
  `from_run_dir`, and `build_activation`/`build_optimizer` str→callable maps.
- `checkpoint.py` — `load_model(key, weights_path, model_config)` (rebuilds
  the exact architecture from a `ModelConfig`), `make_run_dir`.
- `plotting.py` — `plot_m_means`, `plot_learning_curve`.

`scripts/` — entrypoints (`uv run scripts/<x>.py`), each begins with the
device/jax bootstrap:

- `train.py` — `uv run scripts/train.py --config configs/<name>.toml`
  (default `configs/default.toml`). The TOML fully specifies the run; it is
  copied into the run dir as `config.toml` plus a fully-resolved
  `resolved_config.json`. Trains into a fresh `results/<timestamp>/`;
  epoch-0 + every-`checkpoint_every` checkpoints and rollout plots; final
  `weights.eqx` + `learning_curve.png`.
- `evaluate.py` — reads `RUN_DIR`'s `resolved_config.json` to rebuild the
  exact model/data, computes train/val one-step MSE + sp4 rollout
  nRMSE/correlation → `RUN_DIR/metrics.json` + `rollout_nrmse.png`.
- `animate.py` — animates sp4 rollout ⟨m⟩ across `RUN_DIR` checkpoints vs. a
  static ground truth → `RUN_DIR/training.gif`.

Training configs live in `configs/*.toml` (`default.toml` = 10-epoch quick
run; `full.toml` = 256 epochs); copy one per experiment — the file is the
record. `RUN_DIR` is a top-of-file constant in `evaluate.py`/`animate.py`;
point it at the run to evaluate. `main.ipynb` is a scratch notebook.

## Known issues / gotchas


- `README.md` is effectively empty.
