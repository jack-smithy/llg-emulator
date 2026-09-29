# CLAUDE.md

FNO emulator for LLG micromagnetics on permalloy thin films: a one-step map
`m_t -> m_{t+1}` conditioned on the applied field. JAX + Equinox + pdequinox (sibling
checkout at `../pdequinox`). Data is in the Well's HDF5 format, read with
`the_well.data.WellDataset` through a torch DataLoader that collates to numpy.

## Layout

- `main.py` trains one model from CLI args (seed, arch, configuration, dataset,
  hyperparameters) and writes `model.eqx`, `metadata.json` and `stats.json` to
  `results/<dataset>/<arch>/<configuration>/seed_<seed>/`.
- `eval.py` reloads that model, computes one-step MSE/VRMSE on `valid`, rolls out one
  trajectory and saves plots and GIFs beside it.
- `model.py`: `ModelConfig`, `build_model` (pdequinox `ClassicFNO` inside
  `ResidualEmulator`, which adds the predicted increment and renormalises `m`),
  `save_model` / `load_model`.
- `utils.py`: batch layout (`prepare_batch`, `with_coords`, `conditioning`) and the
  jitted `loss_fn`, `update_fn`, `predict`, `rollout`.
- `plot.py`: figures and animations for the eval scripts.
- `physics.py`: neuralmag demag-field module. Not used by the current model.
- `datagen/`: `generate_varied_field.py` (the training-data generator, magnum.np, and
  the home of the simulation constants the others import), `generate_geometries.py`
  (held-out film sizes), `generate_large.py` (one 6144x6144 film),
  `generate_fixed_field.py` (the older SP4 fixed-field dataset) and
  `eval_geometries.py` (rollouts on the geometries). They import each other as
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
conditions on `[Hx, Hy] / Ms` and takes the cell-centre coordinates (in um, from the
sample's `space_grid`) as two extra input channels, which is what lets it run on other
grid sizes.

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
