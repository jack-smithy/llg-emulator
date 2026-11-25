import logging
import sys
import time
from pathlib import Path

import h5py
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
SEED = 0  # with the trajectory index, seeds the generator that draws its field axis
DM_TOL = 10  # [A/m] MinimizerBB convergence tolerance
MAXITER = 2000  # MinimizerBB iterations per field step at most (its default; larger meshes need more)
TIME_BUDGET = 2.75 * 3600  # [s] start no field step that may not finish in this time (3 h jobs)
# random tilt (radians, per cell and component) of m before each field step's minimisation, so a step
# cannot end on a saddle: from a uniform start Barzilai-Borwein can stop on one (0: no kick)
KICK = 0.0
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


def simulate(index, axis, minimizer, external, partial_path, deadline):
    # one quasi-static loop along axis, written step by step to partial_path: m (1, steps, nx, ny, nz, 3)
    # in float32, and the last state in full precision, so a later run resumes exactly where this one
    # stopped; stops before a step that may not finish by deadline (a time.time() value); returns the
    # number of steps done in this run and whether the loop is complete
    field_values = loop(Hmin=Hmin, Hmax=Hmax, steps=steps, axis=axis)
    # start saturated along the first field (+Hmax along axis), or where the last run stopped
    state, _ = setup(n=N, dx=DX, material=material, m0=axis)
    with h5py.File(partial_path, "a") as f:
        if "m" not in f:
            f.create_dataset("m", shape=(1, steps, *N, 3), dtype="float32", chunks=(1, 1, *N, 3))
            f.create_dataset("m_last", shape=(*N, 3), dtype="float64")
            f.attrs["done"] = 0
            f.attrs["n_unconverged"] = 0
        if f.attrs["done"] > 0:
            state.m = torch.as_tensor(f["m_last"][:], dtype=torch.get_default_dtype(), device=state.device)  # type: ignore

        n_done, longest_step = 0, 0.0
        for i in range(f.attrs["done"], steps):
            if time.time() + 1.5 * longest_step > deadline:
                return n_done, False
            start = time.time()
            h_ext = state.Constant(field_values[i])
            external.h = lambda state: h_ext
            if KICK > 0:  # seeded by trajectory and step, so a resumed run gets the same kicks
                noise = np.random.default_rng([SEED, index, i]).normal(scale=KICK, size=(*N, 3))
                m = state.m + torch.as_tensor(noise, dtype=state.m.dtype, device=state.device)  # type: ignore
                state.m = torch.nn.functional.normalize(m, dim=-1)  # type: ignore
            if not minimizer.minimize(state, maxiter=MAXITER, dm_tol=DM_TOL):
                f.attrs["n_unconverged"] += 1
            m = state.m.cpu().numpy()  # type: ignore
            f["m"][0, i] = m
            f["m_last"][...] = m
            f.attrs["done"] = i + 1
            f.flush()
            n_done += 1
            longest_step = max(longest_step, time.time() - start)
    return n_done, True


def main():
    magnumnp.set_log_level(logging.WARNING)  # keep warnings, drop the per-iteration log

    # the terms only depend on the mesh, so they are shared by all trajectories
    # (DemagField computes its kernel on first use and keeps it)
    demag = DemagField()
    exchange = ExchangeField()
    external = ExternalField()
    minimizer = MinimizerBB(terms=[demag, exchange, external])

    deadline = time.time() + TIME_BUDGET
    n_steps_done, n_left, out_of_time = 0, 0, False
    for split, indices in SPLITS.items():
        (OUT_DIR / "data" / split).mkdir(parents=True, exist_ok=True)
        for i in indices:
            path = OUT_DIR / "data" / split / f"shield_{i:02d}.hdf5"
            if path.exists():  # written by an earlier run
                continue
            if out_of_time:  # a trajectory has already stopped at the deadline: start no other
                n_left += 1
                continue

            # per-trajectory generator, so trajectory i always gets the same axis
            axis = random_axis(np.random.default_rng([SEED, i]))
            partial_path = path.with_suffix(".partial.h5")
            start = time.time()
            n_done, finished = simulate(i, axis, minimizer, external, partial_path, deadline)
            n_steps_done += n_done
            if not finished:
                n_left += 1
                out_of_time = True
                continue

            # write to a temporary name first, so a killed job never leaves a partial file
            field_values = loop(Hmin=Hmin, Hmax=Hmax, steps=steps, axis=axis)
            tmp_path = path.with_suffix(".tmp")
            with h5py.File(partial_path) as f:
                n_unconverged = f.attrs["n_unconverged"]
                well_io.write_well_file(  # streams m from the checkpoint frame by frame
                    tmp_path,
                    dx=DX[0],
                    m=f["m"],
                    step_scalars={"Bx": mu_0 * field_values[None, :, 0], "By": mu_0 * field_values[None, :, 1]},
                    trajectory_scalars={"field_angle": [np.arctan2(axis[1], axis[0])]},
                    constants={"Ms": Ms, "A": A},
                )
            tmp_path.rename(path)
            partial_path.unlink()
            print(
                f"trajectory {i} ({split}): axis {np.degrees(np.arctan2(axis[1], axis[0])):.1f} deg, "
                f"{time.time() - start:.0f} s in this run, {n_unconverged} unconverged steps -> {path}",
                flush=True,
            )

    if n_left > 0:
        print(f"{n_left} trajectories left for the next run ({n_steps_done} field steps done in this one)", flush=True)
        # scripts/shield.slrm resubmits on 75: only after progress, so a stuck chain stops
        sys.exit(75 if n_steps_done > 0 else 1)

    if WRITE_STATS and "train" in SPLITS:  # stats.yaml normalises training; test-only sets have none
        well_io.write_stats(OUT_DIR)
    print(f"all trajectories written to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
