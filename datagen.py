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

import well_io

Ms = 1 / mu_0  # (1T)
A = 13e-12
chi = 1e5
Keff = 0.5 * mu_0 * Ms**2
Hmin, Hmax = -200e-3 / mu_0, 200e-3 / mu_0  # (-200mT -> 200mT)
l_ex = (A / Keff) ** 0.5
steps = 201
SEED = 0  # with the trajectory index, seeds the generator that draws its field axis and kicks
DM_TOL = 10  # [A/m] MinimizerBB convergence tolerance
MAXITER = 2000  # MinimizerBB iterations per field step at most (its default; larger meshes need more)
# [rad] random tilt of m (per cell and component) before each field step: the minimiser stops wherever the
# torque is zero, saddle points included, and from a symmetric state nothing pushes it off one
KICK = 0.01
WRITE_STATS = True  # write stats.yaml once the train split is complete (off for parallel workers)

DX = [5e-9 for _ in range(3)]
N = [100, 50, 20]

material = {"Ms": Ms, "A": A}

OUT_DIR = Path("datasets/shield")
SPLITS = {"train": range(24), "valid": range(24, 28), "test": range(28, 32)}  # trajectory indices


def setup(n, dx, material, m0):
    # the whole mesh is the magnet (demag is open-boundary, so no air box is needed)
    mesh = Mesh(n=n, dx=dx)
    state = State(mesh=mesh)

    state.m = state.Constant(m0)  # type: ignore
    state.material.set(material)

    return state, mesh


def random_axis(rng):
    # unit vector along a random axis in the xy plane (the loop sweeps both
    # directions along the axis, so angles in [0, pi) cover every axis)
    angle = rng.uniform(0, np.pi)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


def loop(Hmin, Hmax, steps, axis):
    # array of field values going from
    # Hmax -> Hmin -> Hmax
    # along the unit vector axis
    assert steps % 2 == 1, "steps must be odd"

    down = np.linspace(Hmax, Hmin, steps // 2 + 1)
    up = np.linspace(Hmin, Hmax, steps // 2 + 1)[1:]
    amplitudes = np.concatenate([down, up])

    return amplitudes[:, None] * axis  # (steps, 3)


def simulate(axis, field_values, minimizer, external, rng):
    """One quasi-static hysteresis loop: minimise the energy at each field value in turn.

    Starts saturated along axis; rng draws the kicks. Returns m (steps, nx, ny, nz, 3) and the
    number of field steps whose minimisation did not converge within MAXITER.
    """
    state, _ = setup(n=N, dx=DX, material=material, m0=axis)

    m = np.zeros((len(field_values), *N, 3), dtype=np.float32)
    n_unconverged = 0
    for i, h in enumerate(field_values):
        h_ext = state.Constant(h)
        external.h = lambda state: h_ext

        noise = rng.normal(scale=KICK, size=(*N, 3))
        m_kicked = state.m + torch.as_tensor(noise, dtype=state.m.dtype, device=state.device)  # type: ignore
        state.m = torch.nn.functional.normalize(m_kicked, dim=-1)  # type: ignore

        converged = minimizer.minimize(state, maxiter=MAXITER, dm_tol=DM_TOL)
        if not converged:
            n_unconverged += 1
        m[i] = state.m.cpu().numpy()  # type: ignore
    return m, n_unconverged


def save_trajectory(path, axis, field_values, m):
    """Write one loop as a Well file holding a single trajectory. m: (steps, nx, ny, nz, 3)."""
    well_io.write_well_file(
        path,
        dx=DX[0],
        m=m[None],  # (1, steps, nx, ny, nz, 3)
        step_scalars={"Bx": mu_0 * field_values[None, :, 0], "By": mu_0 * field_values[None, :, 1]},
        trajectory_scalars={"field_angle": [np.arctan2(axis[1], axis[0])]},
        constants={"Ms": Ms, "A": A},
    )


def main():
    magnumnp.set_log_level(logging.WARNING)  # keep warnings, drop the per-iteration log

    # the terms only depend on the mesh, so they are shared by all trajectories
    # (DemagField computes its kernel on first use and keeps it)
    demag = DemagField()
    exchange = ExchangeField()
    external = ExternalField()
    minimizer = MinimizerBB(terms=[demag, exchange, external])

    for split, indices in SPLITS.items():
        (OUT_DIR / "data" / split).mkdir(parents=True, exist_ok=True)
        for i in indices:
            # per-trajectory generator, so trajectory i always gets the same axis and kicks
            rng = np.random.default_rng([SEED, i])
            axis = random_axis(rng)
            field_values = loop(Hmin=Hmin, Hmax=Hmax, steps=steps, axis=axis)  # (steps, 3)

            start = time.time()
            m, n_unconverged = simulate(axis, field_values, minimizer, external, rng)
            path = OUT_DIR / "data" / split / f"shield_{i:02d}.hdf5"
            save_trajectory(path, axis, field_values, m)
            print(
                f"trajectory {i} ({split}): axis {np.degrees(np.arctan2(axis[1], axis[0])):.1f} deg, "
                f"{time.time() - start:.0f} s, {n_unconverged} unconverged steps -> {path}",
                flush=True,
            )

    if WRITE_STATS and "train" in SPLITS:  # stats.yaml normalises training; test-only sets have none
        well_io.write_stats(OUT_DIR)
    print(f"all trajectories written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
