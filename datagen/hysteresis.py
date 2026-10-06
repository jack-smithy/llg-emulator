"""Quasi-static in-plane hysteresis loops of a square test film (1.28 um by default, `--film`): the 5 nm reference solver,
coarse micromagnetics and the closure model on 5 to 80 nm meshes.

    uv run python -m datagen.hysteresis run --runs k1/seed_0 ...    (GPU, JAX)
    uv run python -m datagen.hysteresis plot --runs k1/seed_0 ...   (CPU)

Protocol, per sweep axis (x, y): a uniform state along the axis is relaxed at +H_max
(alpha = 1, 2 ns) on the 5 nm mesh and area-averaged onto each coarse mesh, so every arm
starts from the same state; then the field along the axis ramps linearly
+H_max -> -H_max -> +H_max at `--rate` mT/ns with the other components zero, every arm
stepping 10 ps at a time under the field of that step: `physics.LLGStepper` on the mesh
(the reference at 5 nm, coarse micromagnetics above) and each run's emulator on the same
meshes (5 nm = the native cells): the closure model, solver-in-the-loop or a plain
network, through `utils.model_step`. Recorded: H(t), <m>(t) and a snapshot every
`SNAP_MT` mT.

Written under `RESULTS/<dataset>/hysteresis/`: `solver.npz` (reference and coarse
solvers, run-independent) and `<run with / -> __>.npz`; `plot` draws
`loops_<runs>.png`, one row per sweep axis and one panel per cell size, and prints the
coercive field, remanence and loop area of every arm.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from datagen.eval_scaling import block_mean
from datagen.generate_varied_field import DT, DX, MATERIAL
from paths import RESULTS

MU_0 = 4e-7 * np.pi
FILMS_FINE = {"sq256": 256, "sq512": 512, "sq1024": 1024}  # film -> cells of 5 nm a side
CELLS_NM = (5, 10, 20, 40, 80)
AXES = {"x": 0, "y": 1}
SNAP_MT = 10.0
RELAX_STEPS = 200  # 2 ns at alpha = 1 before the sweep


def field_sequence(h_max_mt, rate_mt_ns, axis):
    """(T, 3) applied field in A/m, 10 ps apart: +H_max -> -H_max -> +H_max along axis."""
    n_half = int(round(2 * h_max_mt / rate_mt_ns / (DT * 1e9)))
    down = np.linspace(h_max_mt, -h_max_mt, n_half, endpoint=False)
    up = np.linspace(-h_max_mt, h_max_mt, n_half, endpoint=False)
    h = np.zeros((2 * n_half, 3))
    h[:, AXES[axis]] = np.concatenate([down, up]) * 1e-3 / MU_0
    return h


def stage_run(out_dir, runs, dataset, h_max, rate, axes, cells, FINE=256, SNAP_MT=SNAP_MT):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from model import load_config, load_model
    from physics import LLGStepper, mesh_physics, stepper_cache
    from utils import model_step

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    Ms, A = MATERIAL["Ms"], MATERIAL["A"]
    l_ex = np.sqrt(2 * A / (MU_0 * Ms**2))
    snap_every = int(round(SNAP_MT / rate / (DT * 1e9)))  # steps per snapshot

    def sweep(step, m0, h_seq, label):
        """Run `step(m, h)` over h_seq (T, 3) in chunks of `snap_every` steps; returns
        bulk <m> (T, 3) and the states at the chunk ends (n_snap, 3, n, n) float16."""

        @eqx.filter_jit
        def chunk(m, hs):
            def body(m, h):
                m1 = step(m, h)
                return m1, m1.mean(axis=(1, 2))

            return jax.lax.scan(body, m, hs)

        bulk, snaps = [], [np.asarray(m0).astype(np.float16)]
        m = m0
        t0 = time.perf_counter()
        for i in range(0, len(h_seq), snap_every):
            m, b = chunk(m, jnp.asarray(h_seq[i : i + snap_every]))
            bulk.append(np.asarray(b))
            snaps.append(np.asarray(m).astype(np.float16))
        print(f"  {label}: {len(h_seq)} steps in {time.perf_counter() - t0:.0f} s, "
              f"<m> end {np.round(bulk[-1][-1], 3)}", flush=True)  # fmt: skip
        return np.concatenate(bulk), np.stack(snaps)

    # the starting states: relaxed at +H_max on the fine mesh, then area-averaged
    relaxer = LLGStepper((FINE, FINE), DX, DT, **(MATERIAL | {"alpha": 1.0}))
    relax_step = eqx.filter_jit(lambda m, h: relaxer(m, h))
    starts = {}
    for axis in axes:
        h0 = jnp.asarray(field_sequence(h_max, rate, axis)[0])
        m = jnp.zeros((3, FINE, FINE)).at[AXES[axis]].set(1.0)
        for _ in range(RELAX_STEPS):
            m = relax_step(m, h0)
        m = np.asarray(m)
        starts[axis] = {nm: block_mean(np.moveaxis(m, 0, -1), int(round(nm / 5))) for nm in cells}
        starts[axis] = {nm: jnp.asarray(np.moveaxis(v, -1, 0)) for nm, v in starts[axis].items()}
        print(f"relaxed start for the {axis} sweep: <m> {np.round(m.mean(axis=(1, 2)), 3)}", flush=True)

    meta = {"h_max_mt": h_max, "rate_mt_ns": rate, "axes": list(axes), "cells_nm": list(cells),
            "snap_mt": SNAP_MT, "dt": DT, "film_um": FINE * DX[0] * 1e6, "fine_cells": FINE}  # fmt: skip
    (out_dir / "config.json").write_text(json.dumps(meta))

    def mesh(nm):
        n = FINE // int(round(nm / 5))
        d = FINE * DX[0] / n
        return n, (d, d, DX[2])

    # the solvers: run-independent, computed once
    solver_file = out_dir / "solver.npz"
    solver_out = dict(np.load(solver_file)) if solver_file.exists() else {}
    for axis in axes:
        h_seq = field_sequence(h_max, rate, axis)
        solver_out[f"H_{axis}"] = h_seq
        for nm in cells:
            if f"bulk_{axis}_{nm:g}" in solver_out:
                continue
            n, dx = mesh(nm)
            solver = stepper_cache((n, n), dx)
            bulk, snaps = sweep(lambda m, h, s=solver: s(m, h), starts[axis][nm], h_seq,
                                f"solver {nm:g} nm, {axis} sweep")  # fmt: skip
            solver_out[f"bulk_{axis}_{nm:g}"], solver_out[f"snap_{axis}_{nm:g}"] = bulk, snaps
            np.savez(solver_file, **solver_out)

    for run in runs:
        path = RESULTS / dataset / run
        config = load_config(path)
        if config.in_frames != 1:
            raise SystemExit(f"{run} needs {config.in_frames} frames of context: no history in a sweep")
        model = eqx.nn.inference_mode(load_model(path, jr.PRNGKey(0), "model"))
        run_file = out_dir / f"{run.replace('/', '__')}.npz"
        run_out = dict(np.load(run_file)) if run_file.exists() else {}
        for axis in axes:
            h_seq = field_sequence(h_max, rate, axis)
            for nm in cells:
                if f"bulk_{axis}_{nm:g}" in run_out:
                    continue
                n, dx = mesh(nm)
                physics = mesh_physics(config, (n, n), dx)
                log_ratio = float(np.log(dx[0] / l_ex))
                x = (jnp.arange(n) + 0.5) * dx[0]
                grid = jnp.stack(jnp.meshgrid(x, x, indexing="ij"), axis=-1)[None]

                # any one-frame emulator through utils.model_step: the closure model,
                # solver-in-the-loop, or a plain network with coordinates
                def step(m, h, physics=physics, log_ratio=log_ratio, grid=grid):
                    cond = jnp.array([[h[0] / Ms, h[1] / Ms, log_ratio]])
                    return model_step(model, m[None, None], cond, grid, h[None], physics)[0]

                bulk, snaps = sweep(step, starts[axis][nm], h_seq, f"{run} {nm:g} nm, {axis} sweep")
                run_out[f"bulk_{axis}_{nm:g}"], run_out[f"snap_{axis}_{nm:g}"] = bulk, snaps
                np.savez(run_file, **run_out)


# --- figures ----------------------------------------------------------------

INK, MUTED, SECOND, GRID = "#0b0b0b", "#8a8984", "#52514e", "#e4e3df"
CLOSURE_COLORS = ("#2a78d6", "#4a3aa7", "#e87ba4", "#eda100")  # slots 1, 7, 5, 4
SOLVER = "#eb6834"


def loop_metrics(h_mt, m):
    """Coercive field (mT, mean of both branches' zero crossings), remanence (|<m>| at
    H = 0 on the descending branch) and the loop area (mT) of one <m_axis>(H) loop."""
    half = len(h_mt) // 2
    out = {}
    for name, sl in (("down", slice(0, half)), ("up", slice(half, None))):
        hh, mm = h_mt[sl], m[sl]
        s = np.where(np.diff(np.sign(mm)) != 0)[0]
        out[f"hc_{name}"] = float(hh[s[0]] - mm[s[0]] * (hh[s[0] + 1] - hh[s[0]]) / (mm[s[0] + 1] - mm[s[0]])) if len(s) else float("nan")
        z = np.argmin(np.abs(hh))
        out[f"mr_{name}"] = float(mm[z])
    out["area"] = float(abs(np.trapezoid(m, h_mt)))
    return out


def stage_plot(out_dir, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = json.loads((out_dir / "config.json").read_text())
    axes, cells = meta["axes"], meta["cells_nm"]
    solver = dict(np.load(out_dir / "solver.npz"))
    arms = {run: dict(np.load(out_dir / f"{run.replace('/', '__')}.npz")) for run in runs}
    fig, axs = plt.subplots(len(axes), len(cells), figsize=(2.9 * len(cells), 2.9 * len(axes) + 0.6),
                            sharex=True, sharey=True)  # fmt: skip
    axs = np.atleast_2d(axs)
    rows = []
    for r, axis in enumerate(axes):
        c = AXES[axis]
        h_mt = solver[f"H_{axis}"][:, c] * MU_0 * 1e3
        ref = solver[f"bulk_{axis}_5"][:, c]
        for col, nm in enumerate(cells):
            ax = axs[r, col]
            ax.plot(h_mt, ref, color=INK, lw=2.0, label="reference (5 nm LLG)")
            if nm != 5:
                m = solver[f"bulk_{axis}_{nm:g}"][:, c]
                ax.plot(h_mt, m, color=SOLVER, lw=1.8, label="LLG Solver")
                rows.append((axis, nm, "LLG Solver", loop_metrics(h_mt, m)))
            for run, color in zip(runs, CLOSURE_COLORS):
                m = arms[run][f"bulk_{axis}_{nm:g}"][:, c]
                ax.plot(h_mt, m, color=color, lw=1.8, label=run if len(runs) > 1 else "emulator")
                rows.append((axis, nm, run, loop_metrics(h_mt, m)))
            if col == 0:
                rows.append((axis, 5, "reference", loop_metrics(h_mt, ref)))
            ax.axhline(0, color=GRID, lw=0.8)
            ax.axvline(0, color=GRID, lw=0.8)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=SECOND, labelsize=8)
            if r == 0:
                ax.set_title(f"{nm:g} nm cells" + (" (native)" if nm == 5 else ""), fontsize=9)
            ax.set_xlabel(f"mu0 H_{axis} (mT)", fontsize=8, color=SECOND)  # each row sweeps its own axis
            ax.tick_params(labelbottom=True)
        axs[r, 0].set_ylabel(f"<m_{axis}>", fontsize=9, color=SECOND)
    handles = {}
    for ax in axs.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"{meta['film_um']:g} um film, in-plane hysteresis at {meta['rate_mt_ns']:g} mT/ns", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    tag = "_".join(r.replace("/", "-") for r in runs)
    path = out_dir / f"loops_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"{'axis':>4} {'cell':>5} {'arm':>24} {'Hc down':>8} {'Hc up':>7} {'Mr down':>8} {'Mr up':>7} {'area':>7}")
    for axis, nm, arm, m in rows:
        print(f"{axis:>4} {nm:>5g} {arm:>24} {m['hc_down']:8.2f} {m['hc_up']:7.2f} {m['mr_down']:8.3f} {m['mr_up']:7.3f} {m['area']:7.1f}")
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("run", "plot"))
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--runs", nargs="*", default=[], help="<configuration>/seed_<s> closure runs")
    p.add_argument("--h-max", type=float, default=50.0, help="mT")
    p.add_argument("--rate", type=float, default=2.5, help="mT/ns")
    p.add_argument("--axes", nargs="+", default=("x", "y"), choices=tuple(AXES))
    p.add_argument("--cells", type=float, nargs="+", default=CELLS_NM)
    p.add_argument("--film", default="sq256", choices=list(FILMS_FINE), help="the square film")
    p.add_argument("--snap-mt", type=float, default=SNAP_MT, help="snapshot every this many mT")
    args = p.parse_args()
    if 5 not in args.cells:
        p.error("--cells must include 5: the 5 nm solver is the reference every loop is drawn against")
    # the 1.28 um film's loops keep their directory; other films get their own
    out_dir = RESULTS / args.dataset / ("hysteresis" if args.film == "sq256" else f"hysteresis_{args.film}")
    if args.stage == "run":
        stage_run(out_dir, args.runs, args.dataset, args.h_max, args.rate, args.axes, args.cells,
                  FILMS_FINE[args.film], args.snap_mt)  # fmt: skip
    else:
        stage_plot(out_dir, args.runs)


if __name__ == "__main__":
    main()
