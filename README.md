# llg-emulator

A learned-closure LLG for permalloy thin films: coarse micromagnetics (exchange, the exact
demag field of the coarse mesh, the applied field) stepped under one extra effective field
that a local ResNet reads off the coarse state. It is trained on magnum.np simulations in
[the Well](https://github.com/PolymathicAI/the_well)'s HDF5 format and evaluated against
magnum.np run directly on the coarse mesh. Built with JAX, Equinox, neuralmag and
[pdequinox](https://github.com/Ceyron/pdequinox) (a sibling checkout at `../pdequinox`).

## Layout

```
llg/              the physics
  constants.py      material, cell size, time step
  physics.py        exchange and demag fields, the LLG stepper, rollouts
  magnum.py         magnum.np initial states, relaxation and simulation
model.py          the closure model; building, saving and loading it
data.py           datasets, films and result paths; loaders, coarse-graining, D4 augmentation
metrics.py        MSE, VRMSE, cosine similarity, hysteresis-loop metrics
plotting.py       the figures
configs/          training recipes
train.py          train one recipe for one seed
evaluate.py       validation, rollouts against magnum.np, hysteresis loops, report, animations
generate.py       the training set and the held-out films
scripts/          Slurm wrappers of the three entry points
```

Every stepper, the LLG solver and the model alike, maps `(m, h) -> m'` on one film
(`m` is `(nx, ny, 3)` and `h` is in A/m), and `physics.rollout` runs any of them.

## Setup

```sh
uv sync
scripts/pull_dataset.sh    # datasets/llg_field_switching from Hugging Face
```

## Train

```sh
sbatch --array=0-1 scripts/train.slrm configs/closure.yaml   # seeds 0 and 1
uv run train.py configs/closure.yaml --seed 0                # the same, in a job of your own
```

A run lands in `results-v2/<dataset>/<recipe>/seed_<seed>/` (`RESULTS_ROOT` sets another
root) with its `config.yaml`, `stats.json`, `model.eqx` and checkpoints. `train.slrm` then
scores it on the validation films, which is what models are selected on.

## Evaluate

The stages share a cache under `results-v2/<dataset>/scaling/`. `truth` and `baseline`
are run once per film; the others take `--runs <recipe>/seed_<s> ...`.

```sh
sbatch -c 4 -t 2:00:00 --wrap "uv run evaluate.py truth --films valid256 sq256 sq512 large"
sbatch scripts/evaluate.slrm baseline --films valid256 sq256 sq512 large
sbatch scripts/evaluate.slrm rollouts --runs closure/seed_0 --films sq256 sq512 large
sbatch scripts/evaluate.slrm hysteresis --runs closure/seed_0 --films sq256
uv run evaluate.py report --runs closure/seed_0 closure/seed_1 --films valid256 --cells 10 20 40
sbatch scripts/evaluate.slrm animate --runs closure/seed_0 --films sq512 --cells 10 20
```

## Generate data

```sh
sbatch --array=0-7 scripts/generate.slrm train --out datasets/llg_field_switching
sbatch -t 16:00:00 scripts/generate.slrm film --film sq1024 --out datasets/llg_field_switching
```

The held-out films are extra files in the dataset's `test` split. The Well loads one
resolution at a time, so `data.FILMS` names each film's file pattern.
