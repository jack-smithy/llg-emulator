"""Datasets in the_well's HDF5 layout, simulated with magnum.np.

    uv run generate.py train --out datasets/llg_field_switching --shard 0 --n-shards 8
    uv run generate.py film --out datasets/llg_field_switching --film sq512

`train` writes data/{train,valid,test}/llg_<split>_<shard>.hdf5 on 256^2 films. `film`
writes one held-out film of FILMS into the test split; its sample j carries the field and
the kind of initial state of 256^2 test sample j.
"""

from argparse import ArgumentParser
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import torch
from jaxtyping import Float

from data import FIELDS, FILMS, N_STEPS, SCALARS
from llg.constants import DT, DX, MATERIAL, MU_0
from llg.magnum import random_state, relax, s_state, simulate, uniform_state

H_MAX = 50e-3  # T, per in-plane component
SP4_FIELD = np.array([-24.6e-3, 4.3e-3, 0.0]) / MU_0
INITIAL_STATES = (random_state, uniform_state, s_state)
TRAIN_CELLS = 256

Field = Float[np.ndarray, "..."]
Sample = tuple[Iterator[Field], Float[np.ndarray, "..."]]


def sample(seed: int, n: int) -> Sample:
    """Trajectory `seed`: its initial state relaxed at zero field, then 1 ns under its
    field, drawn from the seed alone so it does not depend on the split or shard."""
    gen = torch.Generator(device=torch.get_default_device()).manual_seed(seed)
    m0 = INITIAL_STATES[seed % len(INITIAL_STATES)]((n, n), gen)
    hx, hy = np.random.default_rng(seed).uniform(-H_MAX, H_MAX, size=2) / MU_0
    h = np.array([hx, hy, 0.0])
    return trajectory(m0, h), h


def sp4_sample(n: int) -> Sample:
    return trajectory(s_state((n, n)), SP4_FIELD), SP4_FIELD


def trajectory(m0: torch.Tensor, h: Field) -> Iterator[Field]:
    m = relax(m0, DX)
    yield m
    yield from simulate(m, h, DX, N_STEPS)


def write_film(path: Path, samples: list[Sample], n: int):
    """Written under .part, which the well's glob skips, and renamed once complete."""
    part = path.with_name(path.name + ".part")
    with create_file(part, len(samples), n) as f:
        for j, (frames, h) in enumerate(samples):
            print(
                f"{path.name}: sample {j + 1}/{len(samples)}, mu0 H = {h * MU_0 * 1e3} mT",
                flush=True,
            )
            for name, value in zip(SCALARS[3:], h):
                f["scalars"][name][j] = value
            for t, m in enumerate(frames):
                for c, name in enumerate(FIELDS):
                    f["t0_fields"][name][j, t] = m[..., c]
    part.replace(path)


def create_file(path: Path, n_traj: int, n: int) -> h5py.File:
    """An empty the_well file: m as three t0 fields (a rank-1 field must have as many
    components as spatial dims), material constants and the per-sample field as scalars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["dataset_name"] = "micromagnetics_llg_varied_field"
    f.attrs["grid_type"] = "cartesian"
    f.attrs["n_spatial_dims"] = 2
    f.attrs["n_trajectories"] = n_traj
    f.attrs["simulation_parameters"] = list(SCALARS)
    for name, value in MATERIAL.items():
        f.attrs[name] = value

    dims = f.create_group("dimensions")
    dims.attrs["spatial_dims"] = ["x", "y"]
    coords = {
        "time": np.arange(N_STEPS + 1) * DT,
        "x": (np.arange(n) + 0.5) * DX[0],
        "y": (np.arange(n) + 0.5) * DX[1],
    }
    for name, value in coords.items():
        d = dims.create_dataset(name, data=value.astype(np.float32))
        d.attrs["sample_varying"] = False
        d.attrs["time_varying"] = name == "time"

    walls = f.create_group("boundary_conditions")
    for dim in ("x", "y"):
        wall = walls.create_group(f"{dim}_wall")
        wall.attrs["bc_type"] = "WALL"
        wall.attrs["associated_dims"] = [dim]
        wall.attrs["associated_fields"] = list(FIELDS)
        wall.attrs["sample_varying"] = False
        wall.attrs["time_varying"] = False
        wall.create_dataset("mask", data=np.isin(np.arange(n), [0, n - 1]))

    scalars = f.create_group("scalars")
    scalars.attrs["field_names"] = list(SCALARS)
    for name in SCALARS:
        if name in MATERIAL:
            d = scalars.create_dataset(name, data=np.float32(MATERIAL[name]))
        else:
            d = scalars.create_dataset(name, shape=(n_traj,), dtype=np.float32)
        d.attrs["sample_varying"] = name not in MATERIAL
        d.attrs["time_varying"] = False

    fields = f.create_group("t0_fields")
    fields.attrs["field_names"] = list(FIELDS)
    for name in FIELDS:
        d = fields.create_dataset(
            name,
            shape=(n_traj, N_STEPS + 1, n, n),
            dtype=np.float32,
            chunks=(1, 1, n, n),  # the loader reads one frame at a time
            compression="gzip",
            compression_opts=1,
        )
        d.attrs["sample_varying"] = True
        d.attrs["time_varying"] = True
        d.attrs["dim_varying"] = [True, True]
    for group in ("t1_fields", "t2_fields"):
        f.create_group(group).attrs["field_names"] = []
    return f


def generate_train(out: Path, sizes: dict[str, int], seed: int, shard: int, n_shards: int):
    """Trajectory i of the concatenated splits uses seed + i, strided over the shards."""
    first = seed
    for split, size in sizes.items():
        seeds = list(range(first, first + size))[shard::n_shards]
        first += size
        path = out / "data" / split / f"llg_{split}_{shard}.hdf5"
        if seeds and not path.exists():
            write_film(path, [sample(s, TRAIN_CELLS) for s in seeds], TRAIN_CELLS)


def generate_film(out: Path, name: str, n_samples: int, seed: int):
    film = FILMS[name]
    path = out / "data" / film.split / film.pattern
    if path.exists():
        return
    if film.sp4:
        samples = [sp4_sample(film.cells)]
    else:
        samples = [sample(seed + j, film.cells) for j in range(n_samples)]
    write_film(path, samples, film.cells)


def main():
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--n-train", type=int, default=256)
    train.add_argument("--n-valid", type=int, default=32)
    train.add_argument("--n-test", type=int, default=8)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--shard", type=int, default=0)
    train.add_argument("--n-shards", type=int, default=1)
    film = commands.add_parser("film")
    film.add_argument("--out", type=Path, required=True)
    generated = [name for name, f in FILMS.items() if "*" not in f.pattern]
    film.add_argument("--film", required=True, choices=generated)
    film.add_argument("--n-samples", type=int, default=8)
    film.add_argument(
        "--seed", type=int, default=288, help="the first seed of the 256^2 test split"
    )
    args = parser.parse_args()

    if args.command == "train":
        sizes = {"train": args.n_train, "valid": args.n_valid, "test": args.n_test}
        generate_train(args.out, sizes, args.seed, args.shard, args.n_shards)
    else:
        generate_film(args.out, args.film, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
