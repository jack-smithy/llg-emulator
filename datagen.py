"""Quasi-static hysteresis loops of NiFe blocks, one HDF5 file per loop (save_loop), one folder per geometry:

    python datagen.py G2 --split train --n-samples 24

writes datasets/G2/train/loop_00.h5 ... loop_23.h5. Each loop draws its own
field axis and start (draw_config) from a generator seeded by geometry, split and loop index.
"""

import argparse
import logging
import time
from pathlib import Path
import magnumnp
import torch
from magnumnp.common.constants import mu_0
from magnumnp.field_terms import DemagField, ExternalField, ExchangeField
from magnumnp.common import Mesh, State
from magnumnp.solvers import MinimizerBB
import numpy as np

import h5py

Ms = 1 / mu_0  # (1T)
A = 13e-12
chi = 1e5
Keff = 0.5 * mu_0 * Ms**2
Hmin, Hmax = -200e-3 / mu_0, 200e-3 / mu_0  # (-200mT -> 200mT)
l_ex = (A / Keff) ** 0.5
steps = 201

material = {"Ms": Ms, "A": A}

# magnet sizes (um): the paper's shield proportions 5 : 2.5 : 1 and variants (CLUSTER_DATAGEN.md)
geometries_um = {
    "G1": [2, 1, 0.4],
    "G2": [4, 2, 0.8],
    "G3": [4, 1, 0.8],  # narrower
    "G4": [4, 4, 0.8],  # wider
    "G5": [4, 2, 1.6],  # thicker
    "G6": [6, 3, 1.2],  # largest
    "C": [1, 0.5, 0.4],  # the check of 10 nm against 5 nm cells before production
}
splits = ["train", "valid", "test"]


def setup(n, dx, material, m0):
    # the whole mesh is the magnet (demag is open-boundary, so no air box is needed)
    mesh = Mesh(n=n, dx=dx)
    state = State(mesh=mesh)

    # m0: one vector for every cell, or one per cell (nx, ny, nz, 3)
    state.m = state.Constant([0.0, 0.0, 0.0])  # type: ignore
    state.m[...] = torch.as_tensor(m0, device=state.device)  # type: ignore
    state.material.set(material)

    return state, mesh


def random_axis(rng: np.random.Generator):
    # unit vector along a random axis in the xy plane (the loop sweeps both
    # directions along the axis, so angles in [0, pi) cover every axis)
    angle = rng.uniform(0, np.pi)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def random_direction(rng: np.random.Generator):
    # unit vector along a random axis in 3D, uniform on the sphere
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def grid(geometry, dx) -> list[int]:
    # number of cells of size dx [m] along x, y and z
    n = []
    for size in geometries_um[geometry]:
        cells = size * 1e-6 / dx
        assert abs(cells - round(cells)) < 1e-6, f"dx = {dx} m does not divide {size} um"
        n.append(round(cells))
    return n


def draw_config(rng: np.random.Generator, n, p_zero_start, p_oop):
    # field axis, field values and initial magnetisation of one loop on a grid of n cells

    # loop axis
    sweep_axis = random_direction(rng) if rng.random() < p_oop else random_axis(rng)

    # external field values and init magnetization
    if rng.random() < p_zero_start:
        # start from zero with random magnetization
        loop = loop_from_zero(Hmin=Hmin, Hmax=Hmax, steps=steps, axis=sweep_axis)
        m0 = rng.normal(size=(*n, 3))
        m0 = m0 / np.linalg.norm(m0, axis=-1, keepdims=True)
    else:
        # start with magnetization aligned with field and saturated
        loop = loop_from_saturated(Hmin=Hmin, Hmax=Hmax, steps=steps, axis=sweep_axis)
        m0 = sweep_axis  # setup() puts it in every cell

    return {"loop": loop, "axis": sweep_axis, "m0": m0}


def loop_from_saturated(Hmin, Hmax, steps, axis):
    # array of field values going from
    # Hmax -> Hmin -> Hmax
    # along the unit vector axis
    assert steps % 2 == 1, "steps must be odd"

    down = np.linspace(Hmax, Hmin, steps // 2 + 1)
    up = np.linspace(Hmin, Hmax, steps // 2 + 1)[1:]
    amplitudes = np.concatenate([down, up])

    return amplitudes[:, None] * axis  # (steps, 3)


def loop_from_zero(Hmin, Hmax, steps, axis):
    # array of field values going from
    # 0 -> Hmax -> Hmin
    # along the unit vector axis, with the field step of loop(Hmin, Hmax, steps, axis)
    assert Hmin == -Hmax and steps % 4 == 1, "needs a symmetric loop and steps = 4k + 1"

    n_up = (steps - 1) // 4
    up = np.linspace(0, Hmax, n_up + 1)
    down = np.linspace(Hmax, Hmin, 2 * n_up + 1)[1:]
    amplitudes = np.concatenate([up, down])

    return amplitudes[:, None] * axis  # (3 * n_up + 1, 3)


def block_average(m, block):
    # average m (nx, ny, nz, 3) over blocks of block^3 cells: (nx / block, ny / block, nz / block, 3)
    nx, ny, nz, _ = m.shape
    blocks = m.reshape(nx // block, block, ny // block, block, nz // block, block, 3)
    return blocks.mean(dim=(1, 3, 5))


def simulate(
    m0, n, dx, field_values, minimizer, external, material, rng, kick, maxiter, dm_tol, store_block
):
    """One quasi-static hysteresis loop on n cells of size dx [m]: minimise the energy at each field
    value in turn.

    m0 is the start: one unit vector for every cell, or one per cell (nx, ny, nz, 3). Before each
    field step, rng tilts m by kick [rad] per cell and component: the minimiser stops wherever the
    torque is zero, saddle points included, and from a symmetric state nothing pushes it off one.
    Returns m as block averages (steps, nx / store_block, ny / store_block, nz / store_block, 3),
    not renormalised, so coarser averages can still be taken from them exactly, and the number of
    field steps whose minimisation did not converge within maxiter.
    """
    state, _ = setup(n=n, dx=[dx] * 3, material=material, m0=m0)

    n_stored = [k // store_block for k in n]
    m = np.zeros((len(field_values), *n_stored, 3), dtype=np.float32)
    n_unconverged = 0
    for i, h in enumerate(field_values):
        h_ext = state.Constant(h)
        external.h = lambda state: h_ext

        noise = rng.normal(scale=kick, size=(*n, 3))
        m_kicked = state.m + torch.as_tensor(  # type: ignore
            noise,
            dtype=state.m.dtype,  # type: ignore
            device=state.device,
        )
        state.m = torch.nn.functional.normalize(m_kicked, dim=-1)  # type: ignore

        converged = minimizer.minimize(state, maxiter=maxiter, dm_tol=dm_tol)

        if not converged:
            n_unconverged += 1

        m[i] = block_average(state.m, store_block).cpu().numpy()  # type: ignore
    return m, n_unconverged


def save_loop(path, dx, fine_dx, axis, field_values, m):
    """Write one loop as a plain HDF5 file: m (steps, nx, ny, nz, 3), the block averages on stored cells
    of size dx [m] (not renormalised); B (steps, 3), the applied field mu0 H [T]; attributes dx,
    fine_dx (the simulated cell size), Ms, A and the field axis (3,)."""
    with h5py.File(path, "w") as f:
        f["m"] = m
        f["B"] = mu_0 * field_values
        f.attrs.update({"dx": dx, "fine_dx": fine_dx, "Ms": Ms, "A": A, "axis": axis})


def main():
    parser = argparse.ArgumentParser(description="Quasi-static hysteresis loops of one geometry.")
    parser.add_argument("geometry", choices=list(geometries_um))
    parser.add_argument("--split", choices=splits, default="train")
    parser.add_argument("--n-samples", type=int, default=24, help="number of loops")
    parser.add_argument(
        "--first-index",
        type=int,
        default=0,
        help="index of the first loop: parallel workers on one split take disjoint ranges",
    )
    parser.add_argument("--dx", type=float, default=10e-9, help="[m] simulated cell size")
    parser.add_argument(
        "--store-block",
        type=int,
        default=5,
        help="store averages over blocks of this many cells cubed (5: 50 nm at dx 10 nm)",
    )
    parser.add_argument(
        "--maxiter", type=int, default=10000, help="MinimizerBB iterations per field step at most"
    )
    parser.add_argument(
        "--dm-tol", type=float, default=10, help="[A/m] MinimizerBB convergence tolerance"
    )
    parser.add_argument(
        "--kick", type=float, default=0.01, help="[rad] random tilt of m before each field step"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("datasets"), help="writes OUT_DIR/GEOMETRY/SPLIT/loop_NN.h5"
    )
    args = parser.parse_args()

    n = grid(args.geometry, args.dx)
    assert all(k % args.store_block == 0 for k in n), (
        f"--store-block {args.store_block} must divide the grid {n}"
    )
    magnumnp.set_log_level(logging.WARNING)  # keep warnings, drop the per-iteration log

    # one mesh per run, so the terms are shared by all its loops
    # (DemagField computes its kernel on first use and keeps it)
    demag = DemagField()
    exchange = ExchangeField()
    external = ExternalField()
    minimizer = MinimizerBB(terms=[demag, exchange, external])

    out_dir = args.out_dir / args.geometry / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry_number = list(geometries_um).index(args.geometry)
    for i in range(args.first_index, args.first_index + args.n_samples):
        # one generator per loop, so a loop is the same however the indices are shared out over
        # workers, and different for every geometry and split
        rng = np.random.default_rng([args.seed, geometry_number, splits.index(args.split), i])

        cfg = draw_config(rng=rng, n=n, p_zero_start=0.5, p_oop=0.25)

        start_time = time.time()

        m, n_unconverged = simulate(
            m0=cfg["m0"],
            n=n,
            dx=args.dx,
            field_values=cfg["loop"],
            minimizer=minimizer,
            external=external,
            material=material,
            rng=rng,
            kick=args.kick,
            maxiter=args.maxiter,
            dm_tol=args.dm_tol,
            store_block=args.store_block,
        )

        path = out_dir / f"loop_{i:02d}.h5"
        save_loop(
            path,
            dx=args.dx * args.store_block,
            fine_dx=args.dx,
            axis=cfg["axis"],
            field_values=cfg["loop"],
            m=m,
        )
        print(
            f"trajectory {i} ({args.split}): axis {np.degrees(np.arctan2(cfg['axis'][1], cfg['axis'][0])):.1f} deg in plane, "
            f"{np.degrees(np.arcsin(cfg['axis'][2])):.1f} deg out of plane, {len(cfg['loop'])} field steps, "
            f"{time.time() - start_time:.0f} s, {n_unconverged} unconverged steps -> {path}",
            flush=True,
        )

    print(f"all trajectories written to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
