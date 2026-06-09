# llg-emulator

Learn LLG (Landau–Lifshitz–Gilbert) magnetization dynamics for a permalloy-ish
thin film. The model is a neural surrogate that maps one magnetization field +
applied field to the next timestep, then is autoregressively rolled out to
reproduce a full trajectory. Built on JAX / Equinox, using
[PDEquinox](https://github.com/Ceyron/pdequinox) for the backbone architecture
and [neuralmag](https://gitlab.com/neuralmag/neuralmag) for the analytical
demagnetizing field.

After each major change, update this file.

## Stack

- JAX (`jax[cuda13]`), Equinox, Optax, jaxtyping, einops, matplotlib, tqdm.
- `pdequinox` supplies the `ClassicResNet` backbone; `neuralmag` (jax backend)
  computes the demag field.
- `grain` is used for dataloading.
- Python >= 3.13, managed with `uv`. `src/` layout package (hatchling); run
  `uv sync` once, then scripts via `uv run scripts/train.py` etc.
- Each script calls `jax_setup.configure_jax()` at import to persist the JIT
  compilation cache to `.jax_cache/`. The scripts no longer pin a GPU — set
  `CUDA_VISIBLE_DEVICES` in the environment to target a specific device (the
  old `config.DEVICE` constant has been removed).

## Data

External, not in this repo. Lives in a sibling repo at
`../micromagnetic-data/data/dynamics/<size>/{train,val}` (size `small` or
`med`; `config.DATA_ROOT`/`config.dataset_dir`), plus a `sp4/sample_0`
trajectory (`config.SP4_PATH`) used for rollout eval/animation. One directory
per sample (`sample_*`), each containing:
- `m.npy` — magnetization array, shape `(T, W, H, 1, 3)`. The singleton axis
  (a z-slice) is dropped on load and the array is rearranged to channel-first
  `(T, 3, H, W)` float32; trajectories are 100 timesteps.
- `params.json` — includes `H_ext` (3-vector applied field), `n`/`dx` (mesh
  geometry), and `material.Ms` (saturation magnetization). `H_ext` is
  nondimensionalized by dividing by `Ms`.

## Layout

`src/llg_emulator/` — importable library:

- `config.py` — constants only: `JAX_CACHE_DIR`, `DATA_ROOT`, `SIZE`,
  `SP4_PATH`, `RESULTS_DIR`, and `dataset_dir(split, size)`.
- `jax_setup.py` — `configure_jax()` (sets the JIT compilation cache dir).
- `model.py` — `LLGEmulator`, the neural surrogate (a PDEquinox
  `ClassicResNet` backbone + analytical demag input + FiLM-conditioned applied
  field + tangent-space residual update), plus the top-level `FiLM` module.
  Each call concatenates `m_t` (3 ch) with `h_demag = demag(m_t)` (3 ch, a
  `physics.DemagField` submodule recomputed every call/rollout step) into the
  backbone's 6 input channels, so the network only learns the short-range
  terms (exchange/anisotropy) on top of the supplied long-range demag field.
  The constant `H_ext` 3-vector drives the `FiLM` module, which emits
  per-(point, channel) `(gamma, beta)` and modulates the hidden features after
  the lifting layer and after every block (`n_pts = num_blocks + 1`
  modulation points; the FiLM's final layer is zero-init ⇒ identity modulation
  at start, so the model begins as the plain residual path). Residual
  formulation (`step`): the backbone output `dm` is projected onto `m_t`'s
  tangent plane (LLG keeps `|m|=1`), then `m_{t+1} = normalize(m_t + dm_perp)`.
  The `demag` submodule has no trainable params; its tensor `demag.N` must be
  frozen via `training.trainable_filter`.
- `physics.py` — `DemagField` (Equinox module) wraps neuralmag's demag solver:
  at construction it builds a neuralmag `State`/`DemagField` for the mesh
  geometry (`n`, `dx`, `Ms`) and caches the precomputed demag tensor `N` as a
  buffer (no trainable params); `__call__(m)` applies neuralmag's FFT `h_cell`
  convolution. Single channel-first sample `(3, A, B)` → demag field
  `(3, A, B)`, nondimensionalized by `Ms` by default (matching the `H_ext / Ms`
  input convention; pass `nondim=False` for the raw A/m field).
  `from_params(params)` builds it from a sample's params.json. jit/vmap-friendly.
- `data.py` — dataloading. `load_metadata`/`load_trajectory` (one sample:
  channel-first `(T, 3, H, W)` float32 trajectory + its nondimensionalized
  `H_ext`), `load_trajectories` (reads a whole split in parallel via a thread
  pool), `LLGStepperSource` (a `grain.RandomAccessDataSource` that flattens all
  trajectories into consecutive `(m0, m1, H)` pairs without materializing
  windows), and `dataloader_factory(source, batch_size, ...)` returning a
  `seed -> grain.IterDataset` closure that shuffles, prefetches, and batches
  with `drop_remainder=True`.
- `rollout.py` — `rollout(stepper_fn, n, *, include_init=False)` wraps a
  one-step autonomous `stepper_fn` into a `jax.lax.scan` unroll, and
  `rollout_trajectory(model, m_true, H_ext, *, include_init=True)` is the
  one-call helper (used by all three scripts) that closes over `H_ext` and
  rolls out from `m_true[0]`.
- `metrics.py` — `nRMSE`, `correlation`.
- `training.py` — `count_parameters` (trainable only), `trainable_filter`
  (bool pytree freezing the fixed `demag.N`), `loss_fn` (one-step MSE, jitted,
  the single training objective shared by train + evaluate), `update_fn`
  (partitions out the frozen demag tensor before the grad step; jitted with
  donation), `train_epoch`, `val_epoch`.
- `experiment.py` — `TrainConfig` (`seed`, `epochs`, `checkpoint_every` +
  nested `model`/`data`/`optim`/`wandb`) with typed dataclasses: `ModelConfig`
  (`hidden_channels`, `num_blocks`, `activation`, plus the demag mesh geometry
  `mesh_n`/`mesh_dx`/`demag_p`, defaulting to the 256×256×1 film so the demag
  tensor reconstructs exactly on reload), `DataConfig` (`size`, `batch_size`,
  `viz_sample`), `OptimConfig` (`name`, `lr`), `WandbConfig` (`project`,
  `entity`, `mode` [offline|online|disabled, default offline], optional run
  `name`). `from_toml`/`from_dict` (stdlib
  `tomllib`; unknown keys raise), `save` (copies source `.toml` as
  `config.toml` + writes `resolved_config.json`), `from_run_dir`, and
  `build_activation`/`build_optimizer` str→callable maps.
- `checkpoint.py` — `load_model(key, weights_path, model_config)` (rebuilds the
  exact architecture from a `ModelConfig`, defaulting to `ModelConfig()`), and
  `make_run_dir` (fresh timestamped run dir with `checkpoints/weights/` and
  `checkpoints/trjs/` subdirs).
- `plotting.py` — `plot_m_means`, `plot_learning_curve`.

`scripts/` — entrypoints (`uv run scripts/<x>.py`), each calls
`configure_jax()` at import:

- `train.py` — `uv run scripts/train.py --config configs/<name>.toml`
  (default `configs/default.toml`). The TOML fully specifies the run; it is
  copied into the run dir as `config.toml` plus a fully-resolved
  `resolved_config.json`. Trains into a fresh `results/<timestamp>/`; writes an
  epoch-0 checkpoint then every-`checkpoint_every` weights
  (`checkpoints/weights/weights_epoch_N.eqx`) and rollout plots
  (`checkpoints/trjs/trj_epoch_N.png`); final `weights.eqx` +
  `learning_curve.png`. All metrics/plots/config are also logged to Weights &
  Biases via `wandb.init`/`wandb.log` (per-epoch `train_loss`/`val_loss`,
  rollout images, learning curve; param count + dataset sizes in the run
  summary; full `asdict(cfg)` as the wandb config). Defaults to `offline` mode
  (SLURM-friendly) with offline run files written under the run dir — sync
  afterward with `wandb sync <run_dir>/wandb/offline-run-*`.
- `evaluate.py` — reads `RUN_DIR`'s `resolved_config.json` to rebuild the exact
  model/data, computes train/val one-step MSE + sp4 rollout nRMSE/correlation →
  `RUN_DIR/metrics.json` + `rollout_nrmse.png`.
- `animate.py` — animates sp4 rollout ⟨m⟩ across `RUN_DIR` checkpoints vs. a
  static ground truth → `RUN_DIR/training.gif`.

Training configs live in `configs/*.toml`; copy one per experiment — the file
is the record. `default.toml` is the 128-epoch run (`hidden_channels=64`,
`batch_size=50`, `lr=1e-4`, `small` dataset); `full.toml` is the intended
256-epoch `med`-dataset run (currently stale — see below). `RUN_DIR` is a
top-of-file constant in `evaluate.py`/`animate.py`; point it at the run to
evaluate.

## Potential issues / next steps

- `configs/full.toml` is stale and will not load: it carries `num_modes`
  (leftover from the previous FNO backbone) under `[model]` and `warmup_steps`
  under `[data]`, neither of which is a valid config key, so
  `TrainConfig.from_toml` raises `ValueError` on unknown keys. Remove those
  keys (and re-tune `hidden_channels`/`batch_size`/`lr`) before using it.
- `evaluate.py` passes `cfg.data.warmup_steps` (as the `seed` argument to
  `dataset_loss`) but `DataConfig` has no `warmup_steps` field → `AttributeError`.
  Its `RUN_DIR` default also points at a non-existent run
  (`2026-05-26_12-18-10`); `animate.py` shares the same stale default. Both
  need pointing at a real `results/<timestamp>/` dir.
- The training objective is one-step MSE only — there is no multi-step /
  rollout-in-the-loop loss. The orphaned `warmup_steps` config above suggests a
  curriculum/multi-step loss was planned but not implemented.
- The demag mesh geometry (`ModelConfig.mesh_n`/`mesh_dx`, default 256×256×1) is
  fixed at construction and baked into the checkpoint; the model only works at
  that spatial resolution. Confirm the `small`/`med` datasets share it, or make
  the mesh data-derived.
- `diffrax` and `torch` are declared in `pyproject.toml` but not imported by
  `src/`/`scripts/` (torch is likely transitive via neuralmag). Consider
  pruning the direct deps.
- There are no automated tests (`scripts/test.py` was removed).
- `README.md` is effectively empty.
