"""Permalloy LLG trajectories with a varying applied field, in `the_well` HDF5 format.

Runs in the project env: `uv run generate_varied_field.py --out <dir>`.

Sibling of `generate_fixed_field.py`, which holds the applied field at SP4 field 1 and
varies only the initial condition. Here the magnet is the same 256x256x1 cells of
5x5x3 nm permalloy in every sample, and **the applied field varies per trajectory**:
in-plane, each of Hx, Hy uniform on +/-50 mT. Initial conditions cycle through random
per-cell directions, a single random direction, and SP4's s-state with randomised
canting. Material, dt and total time are unchanged.

`m` is a 3-vector on a 2D grid, which cannot be a `t1` field: the well asserts a rank-1
field's trailing dim equals `n_spatial_dims`. So the components go in as three `t0`
scalar fields `mx, my, mz` — the loader concatenates them back into a 3-channel sample in
`field_names` order. The per-sample field arrives as `Hx, Hy, Hz` in
`batch["constant_scalars"]`, after the material constants.

Generation parallelises over a slurm array with `--shard i --n-shards n`: the well globs
a split directory and indexes across files, so each task writes its own
`llg_<split>_<i>.hdf5` and its own partial stats. `--merge-stats` sums the parts into
`stats.yaml` afterwards and is instant.

Layout written (what `WellDataset(path=<out>, well_split_name="train")` expects):

    <out>/data/{train,valid,test}/llg_*.hdf5
    <out>/stats.yaml
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import h5py
import numpy as np
import yaml

N = (256, 256, 1)  # cells; z is the film thickness, so the grid is 2D
DX = (5e-9, 5e-9, 3e-9)
DT = 1e-11
T_TOT = 1e-9
MATERIAL = {"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02}
MU_0 = 4e-7 * math.pi
H_MAX = 50e-3  # T, per in-plane component
SPATIAL_DIMS = ("x", "y")
FIELDS = ("mx", "my", "mz")
H_NAMES = ("Hx", "Hy", "Hz")
N_T = round(T_TOT / DT) + 1


def sample_field(seed):
    """Applied field for trajectory `seed`, in A/m.

    Keyed on the seed alone, so a sample's physics does not depend on which split or
    shard it lands in, or on how many samples precede it.
    """
    rng = np.random.default_rng(seed)
    hx, hy = rng.uniform(-H_MAX, H_MAX, size=2) / MU_0
    return float(hx), float(hy), 0.0


# --- simulation -------------------------------------------------------------


def _rand_unit(shape, gen):
    """Uniform on the sphere (magnum.np's own randM, but with an explicit RNG)."""
    import torch

    theta = 2.0 * torch.pi * torch.rand(shape, generator=gen)
    phi = torch.acos(2.0 * torch.rand(shape, generator=gen) - 1.0)
    return torch.stack(
        [
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
        ],
        dim=-1,
    )


def init_random(state, gen):
    """Per-cell random directions; relaxes into a multi-domain / vortex state."""
    return _rand_unit(state.mesh.n, gen)


def init_uniform(state, gen):
    """A single random direction; relaxes into a near-uniform state."""
    m = state.Constant([0.0, 0.0, 0.0])
    m[...] = _rand_unit((), gen)
    return m


def init_s_state(state, gen):
    """SP4's initial condition — +x with the ends canted — with the cant randomised.

    Amplitude, chirality and how many cells at each end are canted all vary, so the
    relaxed s-states differ between samples instead of repeating one fixed structure.
    """
    import torch

    m = state.Constant([0.0, 0.0, 0.0])
    m[..., 0] = 1.0
    cant = 0.2 + 0.8 * torch.rand((), generator=gen)
    sign = 1.0 if torch.rand((), generator=gen) < 0.5 else -1.0
    w = int(torch.randint(1, 5, (), generator=gen))
    m[:w, :, :, 1] = sign * cant
    m[-w:, :, :, 1] = -sign * cant
    return m


INITS = (init_random, init_uniform, init_s_state)


def simulate(seed, init_fn):
    """Relax at zero field, then integrate T_TOT under this sample's field.

    Returns `(n_t, nx, ny, 3)` float32 — the singleton z axis is squeezed out.
    """
    # imported here so --self-check runs without pulling in torch
    import torch
    from magnumnp import (
        DemagField,
        ExchangeField,
        ExternalField,
        LLGSolver,
        Mesh,
        State,
        normalize,
    )

    gen = torch.Generator(device=torch.get_default_device()).manual_seed(seed)

    state = State(Mesh(N, DX))
    state.material = dict(MATERIAL)
    state.m = normalize(init_fn(state, gen).clone())

    demag, exchange = DemagField(), ExchangeField()
    LLGSolver([demag, exchange]).relax(state)  # relax() forces alpha=1 internally
    normalize(state.m)  # relax() steps without renormalising, unlike step()

    llg = LLGSolver([demag, exchange, ExternalField(list(sample_field(seed)))])
    state.t = 0.0

    frames = np.empty((N_T, N[0], N[1], 3), dtype=np.float32)
    frames[0] = state.m.squeeze(2).cpu().numpy()
    for i in range(1, N_T):
        llg.step(state, DT)
        frames[i] = state.m.squeeze(2).cpu().numpy()
    return frames


# --- the_well writer --------------------------------------------------------


def create_split(path, n_traj, n_t, dataset_name, n=N):
    """Open an HDF5 split with every dataset preallocated; `write_sample` fills it.

    Written incrementally rather than stacked: one trajectory is 79 MB, so a split's
    worth of them is better off going straight to disk. `n` is the grid, for a caller
    writing a different size than this module's (`generate_large.py`).
    """
    const = {k: MATERIAL[k] for k in ("Ms", "A", "alpha")}
    nx, ny = n[0], n[1]

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["dataset_name"] = dataset_name
    f.attrs["grid_type"] = "cartesian"
    f.attrs["n_spatial_dims"] = len(SPATIAL_DIMS)
    f.attrs["n_trajectories"] = n_traj
    f.attrs["simulation_parameters"] = list(const) + list(H_NAMES)
    for name, value in const.items():
        f.attrs[name] = value

    dims = f.create_group("dimensions")
    dims.attrs["spatial_dims"] = list(SPATIAL_DIMS)
    # cell centres, and the physical time of each frame
    coords = {
        "time": np.linspace(0.0, T_TOT, n_t),
        "x": (np.arange(nx) + 0.5) * DX[0],
        "y": (np.arange(ny) + 0.5) * DX[1],
    }
    for name, value in coords.items():
        d = dims.create_dataset(name, data=value.astype(np.float32))
        d.attrs["sample_varying"] = False
        d.attrs["time_varying"] = name == "time"

    # a thin film in vacuum: no periodicity, every outer face is a wall
    bcs = f.create_group("boundary_conditions")
    for dim, size in zip(SPATIAL_DIMS, (nx, ny)):
        bc = bcs.create_group(f"{dim}_wall")
        bc.attrs["bc_type"] = "WALL"
        bc.attrs["associated_dims"] = [dim]
        bc.attrs["associated_fields"] = list(FIELDS)
        bc.attrs["sample_varying"] = False
        bc.attrs["time_varying"] = False
        mask = np.zeros(size, dtype=bool)
        mask[[0, -1]] = True
        bc.create_dataset("mask", data=mask)

    g = f.create_group("scalars")
    g.attrs["field_names"] = list(const) + list(H_NAMES)
    for name, value in const.items():
        d = g.create_dataset(name, data=np.float32(value))  # shape (), constant
        d.attrs["sample_varying"] = False
        d.attrs["time_varying"] = False
    for name in H_NAMES:
        d = g.create_dataset(name, shape=(n_traj,), dtype=np.float32)
        d.attrs["sample_varying"] = True
        d.attrs["time_varying"] = False

    g = f.create_group("t0_fields")
    g.attrs["field_names"] = list(FIELDS)
    for name in FIELDS:
        # the loader reads one (trajectory, frame) at a time; chunk to match
        d = g.create_dataset(
            name,
            shape=(n_traj, n_t, nx, ny),
            dtype=np.float32,
            chunks=(1, 1, nx, ny),
            compression="gzip",
            compression_opts=1,
        )
        d.attrs["sample_varying"] = True
        d.attrs["time_varying"] = True
        d.attrs["dim_varying"] = [True] * len(SPATIAL_DIMS)

    for group in ("t1_fields", "t2_fields"):
        f.create_group(group).attrs["field_names"] = []
    return f


def write_sample(f, j, frames, h_ext):
    """Write one `(n_t, nx, ny, 3)` trajectory into slot `j`."""
    for i, name in enumerate(FIELDS):
        f["t0_fields"][name][j] = frames[..., i]
    for name, value in zip(H_NAMES, h_ext):
        f["scalars"][name][j] = value


# --- stats ------------------------------------------------------------------


def new_acc():
    return {k: {"n": 0.0, "s": np.zeros(3), "s2": np.zeros(3)} for k in ("m", "d")}


def accumulate(acc, frames):
    """Running sums, so stats never need the whole split in memory at once."""
    a = frames.astype(np.float64)
    for key, x in (("m", a), ("d", np.diff(a, axis=0))):
        acc[key]["n"] += x.shape[0] * x.shape[1] * x.shape[2]
        acc[key]["s"] += x.sum(axis=(0, 1, 2))
        acc[key]["s2"] += (x**2).sum(axis=(0, 1, 2))


def write_part(out, shard, acc):
    """This shard's running sums; a shard cannot see the others' samples."""
    out.mkdir(parents=True, exist_ok=True)
    (out / f"stats_part_{shard}.json").write_text(
        json.dumps(
            {
                k: {n: np.asarray(v).tolist() for n, v in d.items()}
                for k, d in acc.items()
            }
        )
    )


def merge_stats(out):
    """Sum every `stats_part_*.json` under `out` into `stats.yaml`.

    Parts are kept, so re-running one shard and merging again is correct. Clear them by
    hand if you shrink --n-train, or the dropped samples still count.
    """
    acc = new_acc()
    parts = sorted(out.glob("stats_part_*.json"))
    assert parts, f"no stats parts in {out}"
    for path in parts:
        for key, d in json.loads(path.read_text()).items():
            for name, value in d.items():
                acc[key][name] += np.asarray(value)
    print(f"merged {len(parts)} stats parts")
    return write_stats(out / "stats.yaml", acc)


def write_stats(path, acc):
    """`stats.yaml` over the train split, keyed by field name (t0 ⇒ plain floats)."""
    stats = {
        k: {} for k in ("mean", "std", "rms", "mean_delta", "std_delta", "rms_delta")
    }
    for i, name in enumerate(FIELDS):
        for key, suffix in (("m", ""), ("d", "_delta")):
            n, s, s2 = acc[key]["n"], acc[key]["s"][i], acc[key]["s2"][i]
            mean, ms = s / n, s2 / n
            stats["mean" + suffix][name] = float(mean)
            stats["std" + suffix][name] = float(math.sqrt(max(ms - mean**2, 0.0)))
            stats["rms" + suffix][name] = float(math.sqrt(ms))
    path.write_text(yaml.safe_dump(stats, default_flow_style=False, sort_keys=True))
    return stats


# --- check ------------------------------------------------------------------


def self_check():
    """Writer and stats-merge round-trip on fake frames. No physics, no GPU."""
    rng = np.random.default_rng(0)
    n_t = 3
    frames = rng.normal(size=(n_t, N[0], N[1], 3)).astype(np.float32)
    acc = new_acc()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        path = out / "data" / "train" / "llg_train_0.hdf5"
        with create_split(path, 1, n_t, "check") as f:
            write_sample(f, 0, frames, (1.0, 2.0, 0.0))
            accumulate(acc, frames)

        with h5py.File(path, "r") as f:
            mx = f["t0_fields"]["mx"][0]
            assert mx.shape == (n_t, N[0], N[1]), mx.shape
            assert np.array_equal(mx, frames[..., 0]), "frames mangled"
            assert f["scalars"]["Hx"][0] == 1.0 and f["scalars"]["Hz"][0] == 0.0
            assert f["dimensions"]["x"].shape == (N[0],)

        # a merge of two shards must equal one accumulator over both their samples
        other = rng.normal(loc=3.0, size=(n_t, N[0], N[1], 3)).astype(np.float32)
        acc2 = new_acc()
        accumulate(acc2, other)
        write_part(out, 0, acc)
        write_part(out, 1, acc2)
        both = np.concatenate([frames, other], axis=0)[..., 0]
        assert np.isclose(merge_stats(out)["mean"]["mx"], both.mean(), atol=1e-6)

        (out / "stats_part_1.json").unlink()
        stats = merge_stats(out)

    assert np.isclose(stats["mean"]["mx"], frames[..., 0].mean(), atol=1e-6)
    assert np.isclose(stats["std"]["mx"], frames[..., 0].std(), atol=1e-6)
    d = np.diff(frames.astype(np.float64), axis=0)[..., 2]
    assert np.isclose(stats["rms_delta"]["mz"], np.sqrt((d**2).mean()), atol=1e-6)
    print("self-check ok")


# --- driver -----------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-valid", type=int, default=8)
    p.add_argument("--n-test", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--merge-stats", action="store_true")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()

    if args.self_check:
        return self_check()
    if args.out is None:
        p.error("--out is required")
    if args.merge_stats:
        return merge_stats(args.out)

    from magnumnp import set_log_level

    set_log_level(250)

    # Trajectory i uses seed base+i whatever split it lands in, so adding samples to one
    # split cannot change another's physics.
    n = (args.n_train, args.n_valid, args.n_test)
    bounds = np.cumsum((0,) + n)
    splits = {
        s: range(int(lo), int(hi))
        for s, lo, hi in zip(("train", "valid", "test"), bounds, bounds[1:])
    }

    acc = new_acc()
    for split, all_indices in splits.items():
        # every sample is the same size now, so a plain stride balances the shards
        indices = all_indices[args.shard :: args.n_shards]
        if not indices:  # more shards than samples in this split
            continue
        path = args.out / "data" / split / f"llg_{split}_{args.shard}.hdf5"
        with create_split(
            path, len(indices), N_T, "micromagnetics_llg_varied_field"
        ) as f:
            for j, i in enumerate(indices):
                init_fn = INITS[i % len(INITS)]
                h_ext = sample_field(args.seed + i)
                print(
                    f"[{split}] sample {j + 1}/{len(indices)} seed={args.seed + i} "
                    f"init={init_fn.__name__} "
                    f"H=({h_ext[0] * MU_0 * 1e3:+.1f}, {h_ext[1] * MU_0 * 1e3:+.1f}) mT",
                    flush=True,
                )
                frames = simulate(args.seed + i, init_fn)
                write_sample(f, j, frames, h_ext)
                if split == "train":
                    accumulate(acc, frames)

    write_part(args.out, args.shard, acc)
    if args.n_shards == 1:
        merge_stats(args.out)
    print(f"wrote {args.out} shard {args.shard}/{args.n_shards}")


if __name__ == "__main__":
    main()
