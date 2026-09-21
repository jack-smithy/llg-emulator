# /// script
# requires-python = ">=3.11"
# dependencies = ["magnumnp>=2.2.0", "h5py>=3.11", "numpy", "pyyaml"]
# ///
"""Permalloy thin-film LLG trajectories in `the_well` HDF5 format.

Self-contained: run it with `uv run --script src/datagen/generate_well.py`, which
builds its own environment (magnum.np drags in torch; keeping it out of the
project's jax env is deliberate). Nothing here imports `llg_emulator`.

Setup is muMAG standard problem #4: 100x25x1 cells of 5x5x3 nm permalloy, the SP4
applied field, 101 frames at dt = 10 ps. **The only thing that varies between
trajectories is the initial condition** — every sample is relaxed at zero field
from a random start, then integrated under the SP4 field. The exact SP4 s-state
initial condition is held out as the `test` split.

Layout written (what `WellDataset(path=<out>, well_split_name="train")` expects):

    <out>/data/{train,valid,test}/llg_*.hdf5
    <out>/stats.yaml

`m` is a 3-vector on a 2D grid, which cannot be a `t1` field: the well asserts a
rank-1 field's trailing dim equals `n_spatial_dims`. So the components go in as
three `t0` scalar fields `mx, my, mz` — the loader concatenates them back into a
3-channel sample in `field_names` order.
"""

import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np
import yaml

N = (100, 25, 1)  # cells; z is the film thickness, so the grid is 2D
DX = (5e-9, 5e-9, 3e-9)
DT = 1e-11
T_TOT = 1e-9
MATERIAL = {"Ms": 8e5, "A": 1.3e-11, "alpha": 0.02}
MU_0 = 4e-7 * math.pi
H_EXT = (-24.6e-3 / MU_0, 4.3e-3 / MU_0, 0.0)  # SP4 field 1
SPATIAL_DIMS = ("x", "y")
FIELDS = ("mx", "my", "mz")


# --- simulation -------------------------------------------------------------


def _rand_unit(shape, gen):
    """Uniform on the sphere (magnum.np's own randM, but with an explicit RNG)."""
    import torch

    theta = 2.0 * torch.pi * torch.rand(shape, generator=gen)
    phi = torch.acos(2.0 * torch.rand(shape, generator=gen) - 1.0)
    return torch.stack(
        [torch.sin(phi) * torch.cos(theta), torch.sin(phi) * torch.sin(theta), torch.cos(phi)],
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


def init_s_state(state, gen=None):
    """The SP4 initial condition: +x with the ends canted, relaxing to the s-state."""
    m = state.Constant([0.0, 0.0, 0.0])
    m[..., 0] = 1.0
    m[0, :, :, 1] = 1.0
    m[-1, :, :, 1] = -1.0
    return m


INITS = (init_random, init_uniform)


def simulate(seed, init_fn):
    """Relax at zero field, then integrate T_TOT under H_EXT.

    Returns `(n_t, nx, ny, 3)` float32 — the singleton z axis is squeezed out.
    """
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

    llg = LLGSolver([demag, exchange, ExternalField(list(H_EXT))])
    state.t = 0.0

    n_t = round(T_TOT / DT) + 1
    frames = np.empty((n_t, N[0], N[1], 3), dtype=np.float32)
    frames[0] = state.m.squeeze(2).cpu().numpy()
    for i in range(1, n_t):
        llg.step(state, DT)
        frames[i] = state.m.squeeze(2).cpu().numpy()
    return frames


# --- the_well writer --------------------------------------------------------


def write_split(path, m, dataset_name):
    """`m` is `(n_traj, n_t, nx, ny, 3)` float32."""
    n_traj, n_t, nx, ny, _ = m.shape
    scalars = {"Ms": MATERIAL["Ms"], "A": MATERIAL["A"], "alpha": MATERIAL["alpha"],
               "Hx": H_EXT[0], "Hy": H_EXT[1], "Hz": H_EXT[2]}

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["dataset_name"] = dataset_name
        f.attrs["grid_type"] = "cartesian"
        f.attrs["n_spatial_dims"] = len(SPATIAL_DIMS)
        f.attrs["n_trajectories"] = n_traj
        f.attrs["simulation_parameters"] = list(scalars)
        for name, value in scalars.items():
            f.attrs[name] = value

        dims = f.create_group("dimensions")
        dims.attrs["spatial_dims"] = list(SPATIAL_DIMS)
        # cell centres, and the physical time of each frame
        coords = {"time": np.linspace(0.0, T_TOT, n_t),
                  "x": (np.arange(nx) + 0.5) * DX[0],
                  "y": (np.arange(ny) + 0.5) * DX[1]}
        for name, value in coords.items():
            d = dims.create_dataset(name, data=value.astype(np.float32))
            d.attrs["sample_varying"] = False
            d.attrs["time_varying"] = name == "time"

        # a thin film in vacuum: no periodicity, every outer face is a wall
        bcs = f.create_group("boundary_conditions")
        for dim, n in zip(SPATIAL_DIMS, (nx, ny)):
            bc = bcs.create_group(f"{dim}_wall")
            bc.attrs["bc_type"] = "WALL"
            bc.attrs["associated_dims"] = [dim]
            bc.attrs["associated_fields"] = list(FIELDS)
            bc.attrs["sample_varying"] = False
            bc.attrs["time_varying"] = False
            mask = np.zeros(n, dtype=bool)
            mask[[0, -1]] = True
            bc.create_dataset("mask", data=mask)

        g = f.create_group("scalars")
        g.attrs["field_names"] = list(scalars)
        for name, value in scalars.items():
            d = g.create_dataset(name, data=np.float32(value))  # shape (), constant
            d.attrs["sample_varying"] = False
            d.attrs["time_varying"] = False

        g = f.create_group("t0_fields")
        g.attrs["field_names"] = list(FIELDS)
        for i, name in enumerate(FIELDS):
            # the loader reads one (trajectory, frame) at a time; chunk to match
            d = g.create_dataset(name, data=np.ascontiguousarray(m[..., i]),
                                 chunks=(1, 1, nx, ny), compression="gzip",
                                 compression_opts=1)
            d.attrs["sample_varying"] = True
            d.attrs["time_varying"] = True
            d.attrs["dim_varying"] = [True] * len(SPATIAL_DIMS)

        for group in ("t1_fields", "t2_fields"):
            f.create_group(group).attrs["field_names"] = []


def write_stats(path, m):
    """`stats.yaml` over the train split, keyed by field name (t0 ⇒ plain floats)."""
    delta = np.diff(m, axis=1)
    stats = {k: {} for k in
             ("mean", "std", "rms", "mean_delta", "std_delta", "rms_delta")}
    for i, name in enumerate(FIELDS):
        a, d = m[..., i].astype(np.float64), delta[..., i].astype(np.float64)
        stats["mean"][name] = float(a.mean())
        stats["std"][name] = float(a.std())
        stats["rms"][name] = float(np.sqrt((a**2).mean()))
        stats["mean_delta"][name] = float(d.mean())
        stats["std_delta"][name] = float(d.std())
        stats["rms_delta"][name] = float(np.sqrt((d**2).mean()))
    path.write_text(yaml.safe_dump(stats, default_flow_style=False, sort_keys=True))
    return stats


def verify(root):
    """Re-read what was written and assert the format invariants the well checks."""
    for split in ("train", "valid", "test"):
        (path,) = sorted((root / "data" / split).glob("*.hdf5"))
        with h5py.File(path, "r") as f:
            n_traj = int(f.attrs["n_trajectories"])
            n_spatial = int(f.attrs["n_spatial_dims"])
            assert n_spatial == len(f["dimensions"].attrs["spatial_dims"])
            for name in f.attrs["simulation_parameters"]:
                assert name in f.attrs and name in f["scalars"]
            assert set(f["scalars"].attrs["field_names"]) == set(f["scalars"])
            n_t = f["dimensions"]["time"].shape[-1]
            shape = (n_traj, n_t) + tuple(
                f["dimensions"][d].shape[-1] for d in f["dimensions"].attrs["spatial_dims"]
            )
            assert list(f["t0_fields"].attrs["field_names"]) == list(FIELDS)
            m = np.stack([f["t0_fields"][name][:] for name in FIELDS], axis=-1)
            assert m.shape[:-1] == shape, (m.shape, shape)
            norm = np.linalg.norm(m, axis=-1)
            assert abs(norm - 1.0).max() < 1e-5, abs(norm - 1.0).max()
        print(f"  {split}: {path.name}  m{m.shape}  ok")
    assert yaml.safe_load((root / "stats.yaml").read_text())["std"].keys() == set(FIELDS)
    print("  stats.yaml ok")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/well/micromagnetics_llg"))
    p.add_argument("--n-train", type=int, default=48)
    p.add_argument("--n-valid", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="0", help="CUDA device id, -1 for CPU")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()

    if args.verify_only:
        verify(args.out)
        return

    os.environ.setdefault("CUDA_DEVICE", args.device)  # magnum.np reads this on import
    from magnumnp import set_log_level

    set_log_level(25)

    # Trajectory i uses seed base+i whatever split it lands in, so adding samples
    # to one split cannot change another.
    splits = {"train": range(args.n_train),
              "valid": range(args.n_train, args.n_train + args.n_valid)}
    train_m = None
    for split, indices in splits.items():
        frames = []
        for i in indices:
            init_fn = INITS[i % len(INITS)]
            print(f"[{split}] sample {len(frames) + 1}/{len(indices)} "
                  f"seed={args.seed + i} init={init_fn.__name__}", flush=True)
            frames.append(simulate(args.seed + i, init_fn))
        m = np.stack(frames)
        write_split(args.out / "data" / split / f"llg_{split}.hdf5", m, "micromagnetics_llg")
        if split == "train":
            train_m = m

    # held out: the exact SP4 initial condition, i.e. the standard problem itself
    m = np.stack([simulate(args.seed, init_s_state)])
    write_split(args.out / "data" / "test" / "llg_test.hdf5", m, "micromagnetics_llg")

    write_stats(args.out / "stats.yaml", train_m)
    print(f"wrote {args.out}")
    verify(args.out)


if __name__ == "__main__":
    main()
