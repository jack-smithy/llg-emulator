# llg-emulator

A coarse-mesh LLG emulator for permalloy thin films: coarse micromagnetics (exchange, the
exact demag field of the coarse mesh, the applied field) corrected by a local ResNet that
reads the coarse state. By default the network corrects each coarse step's output
(solver-in-the-loop, `configs/sil-wide-long.yaml`); the alternative adds a learned
effective field inside the step (the learned closure, `configs/k1-wide-long.yaml`), which
degrades to the plain solver on much coarser meshes but is less accurate at 5-40 nm. It is
trained on magnum.np simulations in
[the Well](https://github.com/PolymathicAI/the_well)'s HDF5 format and evaluated against
magnum.np run directly on the coarse mesh. Built with JAX, Equinox, neuralmag and
[pdequinox](https://github.com/Ceyron/pdequinox) (a sibling checkout at `../pdequinox`).

## Layout

```
llg/              the physics
  constants.py      material, cell size, time step
  physics.py        exchange and demag fields, the LLG stepper, rollouts
  magnum.py         magnum.np initial states, relaxation and simulation
  bulk.py           3D demag, exchange, energy minimisers and exact stray fields (the shield)
model.py          solver-in-the-loop and the learned closures; building, saving, loading
data.py           datasets, films and result paths; loaders, coarse-graining, D4 augmentation
metrics.py        MSE, VRMSE, cosine similarity, hysteresis-loop metrics
plotting.py       the figures
configs/          training recipes
train.py          train one recipe for one seed
evaluate.py       validation, rollouts against magnum.np, hysteresis loops, report, animations
generate.py       the training set and the held-out films
shield.py         the shield study
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
scripts/chain.sh configs/sil-wide-long.yaml 0 3 --gres=gpu:full:1 -c 8   # seed 0, three <= 3 h links
sbatch --array=0-1 scripts/train.slrm configs/aeff-tendency.yaml          # short recipes: one job per seed
uv run train.py configs/sil-wide-long.yaml --seed 0          # in a job of your own
```

A run resumes from its last checkpoint (`state.eqx`), so long recipes run as chains of short
jobs; `--max-minutes` makes a link stop at a checkpoint in time.

Recipes set the formulation (`sil`, `closure`, `energy`: the closure as the gradient of a
learned local energy of m, `aeff`: a fitted exchange constant A_eff(eps)) and the objective:
`rollout` (unrolled rollouts against the coarse-grained truth) or `tendency` (closure
formulations: the coarse dm/dt against the block mean of the 5 nm dm/dt, single snapshots).

A run lands in `results-v2/<dataset>/<recipe>/seed_<seed>/` (`RESULTS_ROOT` sets another
root) with its `config.yaml`, `stats.json`, `model.eqx` and checkpoints. `train.slrm` then
scores it on the validation films, which is what models are selected on.

## Evaluate

The stages share a cache under `results-v2/<dataset>/scaling/`. `truth` and `baseline`
are run once per film; the others take `--runs <recipe>/seed_<s> ...`.

```sh
sbatch -c 4 -t 2:00:00 --wrap "uv run evaluate.py truth --films valid256 sq256 sq512 large"
sbatch scripts/evaluate.slrm baseline --films valid256 sq256 sq512 large
sbatch scripts/evaluate.slrm rollouts --runs sil-wide-long/seed_0 --films sq256 sq512 large
sbatch scripts/evaluate.slrm hysteresis --runs sil-wide-long/seed_0 --films sq256
uv run evaluate.py report --runs sil-wide-long/seed_0 sil-wide-long/seed_1 --films valid256 --cells 10 20 40
sbatch scripts/evaluate.slrm animate --runs sil-wide-long/seed_0 --films sq512 --cells 10 20
```

`report` scores each run against the coarse solver twice: on the pointwise rollout MSE and
on the MSE of the film average ⟨m⟩(t). Switching is chaotic, so the pointwise error of a
1 ns rollout is capped for any model, while the film average, what a sensor or a
hysteresis loop measures, stays predictable. `baseline` and `rollouts` redo files that lack
a metric added since they were written.

## The shield study

`shield.py` takes the 100 x 50 x 20 um NiFe shield of D. Suess's TMR study at 50 mT on a 3D
mesh (`llg/bulk.py`): the soft-magnet (Maxwell, chi -> infinity) limit, coarse micromagnetics,
and the study's coarse-grained domain solver (micromagnetics with A_eff = sigma dx / 2), with
the stray field at the probe points evaluated exactly.

```sh
sbatch --gres=gpu:a100:1 scripts/shield.slrm      # 2, 1, 0.5 um cells, both sigmas, seeds
uv run shield.py report
```

## Generate data

```sh
sbatch --array=0-7 scripts/generate.slrm train --out datasets/llg_field_switching
sbatch -t 16:00:00 scripts/generate.slrm film --film sq1024 --out datasets/llg_field_switching
```

The held-out films are extra files in the dataset's `test` split. The Well loads one
resolution at a time, so `data.FILMS` names each film's file pattern.
