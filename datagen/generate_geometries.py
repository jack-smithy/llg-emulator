"""Held-out film geometries for an emulator trained on 256x256 films.

Runs in the project env, from the repo root:
`uv run python -m datagen.generate_geometries --out datasets/llg_field_switching --geometry sq512`.
`scripts/generate_geometries.slrm` runs one array task per entry of `GEOMETRIES`.

Same 5x5x3 nm permalloy cells, material, dt, 1 ns, applied-field distribution and
initial conditions as `generate_varied_field.py`; **only the film's size and aspect
ratio change.** Sample j of every geometry uses seed `--seed + j`, with the default
`--seed 288` being the first seed of the 256x256 test split, so it carries the same
applied field and the same kind of initial condition as test sample j of the training
dataset. Where the initial condition is drawn per cell, the draw differs with the grid,
but its family does not. 256x256 itself is covered by that test split, and 6144x6144 by
`generate_large.py`.

Every geometry is one file in the `test` split of the training dataset, next to its
256x256 shards `llg_test_<i>.hdf5`:

    <out>/data/test/llg_test_<geometry>.hdf5

The well requires every file it loads from a split to share one spatial resolution, so
load one geometry with its filename as a filter, `WellDataset(path=out,
well_split_name="test", include_filters=[f"llg_test_{geometry}.hdf5"],
use_normalization=False)`; `datagen/eval_geometries.py` runs a trained model on all of
them. The network is fully convolutional and reads the cell coordinates from the sample,
so the 256x256 weights run on any grid unchanged.

`--geometry` also takes an ad-hoc `NXxNY`, e.g. `--geometry 300x200`. The file is written
under a `.part` suffix, which the well's glob ignores, and renamed once complete; a
geometry whose file already exists is skipped, so re-running the array after a failure
only redoes what is missing.
"""

import argparse
import time
from pathlib import Path

from datagen.generate_varied_field import (
    INITS,
    MU_0,
    N_T,
    create_split,
    sample_field,
    simulate,
    write_sample,
)

# cells; every film is 5 nm x 5 nm x 3 nm cells like the training set
GEOMETRIES = {
    "sq64": (64, 64),
    "sq128": (128, 128),
    "sq512": (512, 512),
    "sq1024": (1024, 1024),
    "rect512x128": (512, 128),
    "rect128x512": (128, 512),
    "rect1024x64": (1024, 64),
    "sp4": (100, 25),  # muMAG standard problem 4's 500 x 125 nm element
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--geometry", required=True, help="a GEOMETRIES key or NXxNY")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=288)
    args = p.parse_args()

    nx, ny = GEOMETRIES.get(args.geometry) or map(int, args.geometry.split("x"))
    n = (nx, ny, 1)
    path = args.out / "data" / "test" / f"llg_test_{args.geometry}.hdf5"
    if path.exists():
        print(f"{path} exists, skipping")
        return

    from magnumnp import set_log_level

    set_log_level(250)

    part = path.with_name(path.name + ".part")
    with create_split(
        part, args.n_samples, N_T, "micromagnetics_llg_varied_field", n=n
    ) as f:
        for j in range(args.n_samples):
            seed = args.seed + j
            init_fn = INITS[seed % len(INITS)]  # as generate_varied_field with --seed 0
            h_ext = sample_field(seed)
            print(
                f"[{args.geometry} {nx}x{ny}] sample {j + 1}/{args.n_samples} "
                f"seed={seed} init={init_fn.__name__} "
                f"H=({h_ext[0] * MU_0 * 1e3:+.1f}, {h_ext[1] * MU_0 * 1e3:+.1f}) mT",
                flush=True,
            )
            t0 = time.perf_counter()
            frames = simulate(seed, init_fn, n=n)
            write_sample(f, j, frames, h_ext)
            print(f"  {time.perf_counter() - t0:.0f} s", flush=True)
    part.replace(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
