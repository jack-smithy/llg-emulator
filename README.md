# llg-emulator

Neural emulator for Landau-Lifshitz-Gilbert (LLG) micromagnetics: a dilated ResNet
(no GroupNorm, so strictly local) that steps the magnetisation of a permalloy thin film forward in time, trained
on magnum.np simulations stored in [the Well](https://github.com/PolymathicAI/the_well)'s
HDF5 format. Built with JAX, Equinox and [pdequinox](https://github.com/Ceyron/pdequinox)
(expected as a sibling checkout at `../pdequinox`).

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
