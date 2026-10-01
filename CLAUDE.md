# CLAUDE.md

Neural emulator for LLG micromagnetics on permalloy thin films: a one-step map
`m_t -> m_{t+1}` conditioned on the applied field. JAX + Equinox + pdequinox (sibling
checkout at `../pdequinox`). Data is in the Well's HDF5 format, read with
`the_well.data.WellDataset` through a torch DataLoader; `prepare_batch` turns the
torch batches into numpy views (the eval scripts still collate to numpy directly).

## Layout

- `main.py` trains one model from CLI args (seed, arch, configuration, dataset,
  hyperparameters) for `--num-steps` gradient steps over a continuously cycling
  loader — no epochs, which stopped meaning anything once every batch draws its
  own pool factor, and no in-training validation (that is `eval.py`'s job) — and
  writes `model.eqx`, `metadata.json` and `stats.json` (train loss per 100-step
  window) to `results/<dataset>/[<group>/]<arch>/<configuration>/seed_<seed>/`
  (`--group`; the overnight 2026-09-30 runs are in `claude-experiments/`).
- `eval.py` reloads that model, computes one-step MSE/VRMSE on `valid`, rolls out one
  trajectory and saves plots and GIFs beside it.
- `test.py` smoke-tests the two: one training batch + the eval, via their own CLIs
  (`--results-root`, `--max-batches`), in a temp dir deleted on exit. `make test`
  submits `scripts/test.slrm`.
- `model.py`: `ModelConfig`, `build_model` (pdequinox `ClassicFNO` or `DilatedResNet`
  — `config.arch` — inside `ResidualEmulator`, which adds the predicted increment and
  renormalises `m`), `save_model` / `load_model`. The FNO's spectral weights are
  indexed by mode number on the domain, so they are only physically scaled right on
  films the size of the training set; the dilated ResNet's receptive field is fixed in
  cells, which is what transfers across geometries. `use_norm` keeps pdequinox's
  default GroupNorm, which normalises over the whole film and so makes the network
  non-local; `--no-use-norm` makes it strictly local. `use_solver` makes a hybrid: the
  input starts with `physics.CoarseLLG`'s one-frame prediction S(m), which the network
  output (zero-initialised) is added to, i.e. a learned correction to micromagnetics
  on the model's own mesh.
- `utils.py`: batch layout (`prepare_batch`, `prepare_unrolled`, `spatial_features`,
  `model_inputs`, `conditioning`, `downsample`) and the jitted `loss_fn`, `update_fn`,
  `unrolled_loss_fn` / `update_unrolled_fn` (`--unroll K`), `predict`, `rollout`.
- `plot.py`: figures and animations for the eval scripts.
- `physics.py`: neuralmag demag-field module, verified against magnum.np to ~1e-7;
  `demag_cache(n, dx)` serves the per-mesh module behind `use_demag`. `CoarseLLG` is
  one 10 ps LLG frame computed by neuralmag (its field terms and LLG right-hand side
  via `state.resolve`, diffrax Dopri5 as in `LLGSolverJAX`) on the data's cell grid,
  with neuralmag's nodes placed at the cell centres; a pure function so it can be
  vmapped and differentiated. `llg_cache(n, dx)` serves one per mesh.
- `datagen/`: `generate_varied_field.py` (the training-data generator, magnum.np, and
  the home of the simulation constants the others import), `generate_geometries.py`
  (held-out film sizes), `generate_large.py` (one 6144x6144 film),
  `generate_fixed_field.py` (the older SP4 fixed-field dataset) and
  `eval_geometries.py` (rollouts on the geometries) and `eval_scaling.py` (stages
  truth / baseline / speed / solver / model / demo / report: rollout accuracy and cost
  across cell and film sizes against micromagnetics on the coarse mesh, figures in
  `results/<dataset>/<group>/scaling/report/`). They import each other as
  `datagen.<name>` and the top-level modules directly, so run them from the repo
  root as `uv run python -m datagen.<name>`.
- `scripts/`: Slurm jobs (`*.slrm`, one MIG slice each, knobs as shell variables at the
  top) plus Hugging Face push/pull. `Makefile` wraps the common `sbatch` calls and
  creates `logs/`, which Slurm needs before it can open a job's output file.

## Data

A dataset is a Well root: `datasets/<name>/data/{train,valid,test}/llg_*.hdf5` plus
`stats.yaml`. `m` is stored as three scalar `t0` fields `mx, my, mz` (the Well cannot
hold a 3-vector on a 2D grid), so samples arrive as `(B, T, Lx, Ly, 3)` channel-last and
`prepare_batch` makes them channel-first. Per-sample constants come in
`constant_scalars` in the order `SCALARS = (Ms, A, alpha, Hx, Hy, Hz)`; the model
conditions on `[Hx, Hy] / Ms` (scaled to O(1)), plus `log(delta / l_ex)` with
`--cell-size-cond`, and takes two geometry channels from the sample's `space_grid`,
`--coords`: `absolute` cell-centre coordinates in um (the default), `edge` distances
to the film edges clipped at 0.32 um (bounded on any film size), or `none`. Absolute
coordinates do not transfer to films larger than the training one (2026-09-30
experiments); `edge` + `--cell-size-cond` carries the same information without the
offset. Pooling (`utils.downsample`) cell-averages m (conservative remapping, a block
mean at integer factors; renormalised) onto `k`-times larger cells over the same
film — `k` is any real >= 1 (never finer than the simulation cell), the mesh rounds to
whole cells (`utils.resampled_cells` / `resampled_mesh`) and the grid is rebuilt
analytically — with the demag module built for the resampled mesh.
`main.py --pool-factors k1 k2 ...` draws a factor per training batch (val is scored per
factor; each distinct factor costs one demag build and one jit compile, so pass a
finite list); `eval.py --pool-factor k` evaluates at one cell size;
`eval_geometries --pool-factors ...` sweeps every geometry across the factors, keying
entries `<name>-k<f>` (unsuffixed at factor 1). Factors are recorded in stats.json, not
metadata.json — evals must be told them. NB absolute coordinates leave the
training range on films larger than the training one (`--coords edge` is the fix).
With `use_demag` (the default in `main.py`)
the exact nondimensional demag field of the current state is fed as three more input
channels, recomputed every rollout step, so the network only learns the short-range
physics and the weights transfer to other film sizes.

`datasets/llg_field_switching` is the current dataset: 256x256 cells of 5x5x3 nm
permalloy, random in-plane field up to 50 mT, 101 frames at 10 ps. `geometries/<name>`
holds one Well root per held-out film size and `large/` the 6144x6144 trajectory.
`datasets/`, `results/` and `logs/` are gitignored; the dataset is mirrored on
Hugging Face via `scripts/push_dataset.sh`.

## Conventions

- Run everything with `uv run`; the JAX compilation cache lives in `.jax_cache`.
- `save_model` writes `metadata.json` next to the weights and `load_model` rebuilds the
  skeleton from it; keep the two in sync when adding config fields.
- `main.py` asserts the dataset's `constant_scalar_names` equal `SCALARS`, so new
  generators must write the scalars in that order.
- Lint and format with `ruff` (config in `pyproject.toml`).
- When you make a change which modifies the API, make sure to update ALL python files, slurm scripts, and makefile targets which are affected.