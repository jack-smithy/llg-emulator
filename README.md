# llg-emulator

Neural emulator for Landau-Lifshitz-Gilbert (LLG) micromagnetics: a dilated ResNet
(no GroupNorm, so strictly local) that steps the magnetisation of a permalloy thin film forward in time, trained
on magnum.np simulations stored in [the Well](https://github.com/PolymathicAI/the_well)'s
HDF5 format. Built with JAX, Equinox and [pdequinox](https://github.com/Ceyron/pdequinox)
(expected as a sibling checkout at `../pdequinox`).

## Models

`model.py` builds one of three emulators of a 10 ps frame, all strictly local and all
run on the mesh of whatever film they are given (`utils.model_step` arranges the inputs):

- the plain network (`main.py` defaults): m' = normalise(m + N(m, coords, ...)), with
  `--use-demag` (the exact demag field as inputs) and `--in-frames` (frames of context);
- solver-in-the-loop (`--solver-in-the-loop`, Um et al. 2020): coarse micromagnetics
  (`physics.LLGStepper`) steps m and the network corrects its output;
- the learned-closure LLG (`--closure`): coarse micromagnetics steps m under one extra
  effective-field term, a closure the network reads off the coarse state,
  `H_theta = eps Ms N(m, h_ex / Ms, h_d / Ms; H_ext / Ms, eps)` with `eps = l_ex / delta`.
  Every input is a field in units of Ms on the coarse mesh and the closure is scaled
  by `eps`, so coarsening the mesh moves the network's inputs into the interior of its
  training range and sends the model back to coarse micromagnetics, rather than
  extrapolating a cell-size input; `|m| = 1` holds exactly and the step stays a damped
  precession, so rollouts cannot run away. The `model.py` docstring has the argument.
  Train it with `make train-closure` (`scripts/train_closure_array.slrm`: a kernel-3
  ResNet, `--arch classic`, 8-step unrolls, pool factors 2 4 8, D4 augmentation).

`--pool-factors` trains on the film coarse-grained by integer factors (one drawn per
batch, `utils.downsample`), `--unroll n` on n-step autoregressive windows, and
`--augment` (coordinate-free models only) transforms each batch by a random symmetry
of the square film (`utils.d4`).

## Setup

```sh
uv sync                    # pdequinox is resolved from ../pdequinox
scripts/pull_dataset.sh    # datasets/llg_field_switching from Hugging Face
```

## Train and evaluate

Slurm jobs live in `scripts/` (one MIG slice each; the experiment knobs are shell
variables at the top of each file). The Makefile wraps the common ones:

```sh
make train            # scripts/train.slrm: main.py, then eval.py, for one seed
make train-array      # four seeds in parallel
make eval             # re-evaluate a trained model on the validation split
make eval-geometries  # roll a trained model out on the held-out film geometries
```

The same thing by hand, in the project env:

```sh
uv run main.py --seed 0 --configuration test --dataset llg_field_switching
uv run eval.py --seed 0 --configuration test --dataset llg_field_switching
```

Weights, metrics and plots land in `results-v2/<dataset>/<configuration>/seed_<seed>/`.

## Data generation

The magnum.np generators live in `datagen/`: `generate_varied_field.py` writes the
training dataset (256x256 films with a random in-plane applied field;
`scripts/generate_varied_array.slrm` runs one shard per array task),
`generate_geometries.py` and `generate_large.py` write the held-out films, and
`eval_geometries.py` rolls a trained model out on them. The held-out films are extra
files in the dataset's `test` split, `data/test/llg_test_<name>.hdf5` beside the 256x256
shards `llg_test_<i>.hdf5`; the well loads one resolution at a time, so select a film
with `include_filters=["llg_test_<name>.hdf5"]` (or the shards by excluding the named
files), as `eval_geometries.filters` does. They import the top-level
modules, so run them from the repo root as modules:

```sh
uv run python -m datagen.generate_varied_field --out datasets/<name> --n-train 64
uv run python -m datagen.generate_geometries --out datasets/llg_field_switching --geometry sq512
uv run python -m datagen.eval_geometries --seed 0 --configuration test --dataset llg_field_switching
```

`datagen/eval_scaling.py` measures rollout accuracy and cost across cell and film sizes
against magnum.np run directly on the coarse mesh, in stages sharing a cache under
`results-v2/<dataset>/scaling/` (`truth` first, as a CPU job since it reads the 31 GB
6144^2 film; `report` last; the GPU stages via `scripts/eval_scaling.slrm`):

```sh
sbatch -t 2:00:00 -c 4 --mem=24G --wrap "uv run python -m datagen.eval_scaling truth"
sbatch scripts/eval_scaling.slrm   # STAGE=baseline, speed, model or demo
uv run python -m datagen.eval_scaling report --runs <configuration>/seed_<s> ...
```
