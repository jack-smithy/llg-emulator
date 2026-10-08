"""Evaluation of trained runs against magnum.np, in stages sharing RESULTS/<dataset>/scaling/.

uv run evaluate.py validate --runs sil-wide-long/seed_0
uv run evaluate.py truth --films sq256 valid256      (CPU: the coarse-grained reference)
uv run evaluate.py baseline --films sq256 valid256   (GPU: magnum.np on the coarse mesh)
uv run evaluate.py rollouts --runs sil-wide-long/seed_0 --films sq256 valid256
uv run evaluate.py hysteresis --runs sil-wide-long/seed_0 --films sq256
uv run evaluate.py report --runs sil-wide-long/seed_0 sil-wide-long/seed_1 --films valid256
uv run evaluate.py animate --runs sil-wide-long/seed_0 --films sq512 --cells 10 20
"""

import json
import os
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

import metrics
import plotting
from data import (
    FILMS,
    N_STEPS,
    RESULTS,
    Film,
    cache_dir,
    coarse_grain,
    read_trajectories,
    run_dir,
    to_batch,
    well_loader,
)
from llg.constants import DT, DX, MU_0, coarse_dx
from llg.physics import llg_solver, rollout, rollout_batch
from model import load_model

CELL_NM = DX[0] * 1e9
HYSTERESIS_CELLS = (5, 10, 20, 40, 80)
AXES = {"x": 0, "y": 1}
H_MAX_MT = 50.0
RAMP_MT_PER_NS = 2.5
RELAX_STEPS = 200  # 2 ns at alpha = 1 before the sweep
SWEEP_CHUNK = 400  # steps per jitted chunk of a hysteresis sweep

Field = Float[Array, "..."]


def factors(film: Film, cells: list[float] | None) -> list[int]:
    return [k for k in film.factors if cells is None or CELL_NM * k in cells]


def load_truth(dataset: str, film: str, k: int) -> tuple[Field, Field, tuple[float, ...]]:
    """(n_traj, 1 + N_STEPS, nx, ny, 3), the applied fields and the cell size.

    Stored in float16 except the first frame: its rounding alone moves a switching
    trajectory by up to 1e-2 MSE.
    """
    cache = cache_dir(dataset)
    truth = jnp.asarray(np.load(cache / f"truth_{film}_k{k}.npy"), jnp.float32)
    truth = truth.at[:, 0].set(np.load(cache / f"first_{film}_k{k}.npy"))
    meta = json.loads((cache / f"truth_{film}_k{k}.json").read_text())
    return truth, jnp.asarray(meta["H"], jnp.float32), tuple(meta["dx"])


def save_metrics(path: Path, pred: Field, truth: Field) -> dict[str, Field]:
    """Each metric per rollout step, averaged over the trajectories."""
    scores = {name: fn(pred, truth).mean(axis=0) for name, fn in metrics.METRICS.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **scores)
    return scores


def validate(dataset: str, runs: list[str]):
    """One-step metrics on the validation split, and the learning curve."""
    loader = well_loader(dataset, "valid", n_steps=1, batch_size=16)
    nx, ny = loader.dataset.metadata.spatial_resolution  # type: ignore
    solver = llg_solver((nx, ny), DX)
    for run in runs:
        path = run_dir(dataset, run)
        step = eqx.Partial(load_model(path), solver=solver)
        batches = [to_batch(sample) for sample in loader]
        pred = jnp.concatenate([rollout_batch(step, b.m, b.h, 1)[:, 0] for b in batches])
        truth = jnp.concatenate([b.targets[:, 0] for b in batches])
        stats = json.loads((path / "stats.json").read_text())
        stats["metrics"] = {
            name: float(fn(pred, truth).mean()) for name, fn in metrics.METRICS.items()
        }
        (path / "stats.json").write_text(json.dumps(stats))
        print(f"{run}: {stats['metrics']}")
        plotting.save(plotting.learning_curve(stats), path / "learning_curve.png")


def truth(dataset: str, films: list[str]):
    """Each film's trajectories coarse-grained onto every evaluation mesh."""
    cache = cache_dir(dataset)
    cache.mkdir(parents=True, exist_ok=True)
    for name in films:
        film = FILMS[name]
        todo = [k for k in film.factors if not (cache / f"truth_{name}_k{k}.npy").exists()]
        frames, firsts, hs = defaultdict(list), defaultdict(list), []
        for read, h in read_trajectories(film.files(dataset)):
            trajectory = defaultdict(list)
            for t in range(N_STEPS + 1):
                frame = read(t)
                for k in todo:
                    coarse = coarse_grain(frame, k)
                    trajectory[k].append(coarse.astype(jnp.float16))
                    if t == 0:
                        firsts[k].append(coarse)
            for k in todo:
                frames[k].append(jnp.stack(trajectory[k]))
            hs.append(h.tolist())
        for k in todo:
            np.save(cache / f"truth_{name}_k{k}.npy", jnp.stack(frames[k]))
            np.save(cache / f"first_{name}_k{k}.npy", jnp.stack(firsts[k]))
            meta = {"H": hs, "dx": coarse_dx(k), "fine_cells": [film.cells, film.cells]}
            (cache / f"truth_{name}_k{k}.json").write_text(json.dumps(meta))
            print(f"truth {name} k={k} done", flush=True)


def cases(films: list[str], cells: list[float] | None):
    for name in films:
        for k in factors(FILMS[name], cells):
            yield name, k


def magnum_rollout(reference: Field, hs: Field, dx: tuple[float, ...]) -> Field:
    from llg.magnum import simulate

    trajectories = [
        jnp.stack([jnp.asarray(m, jnp.float32) for m in simulate(m0, h, dx, N_STEPS)])
        for m0, h in zip(np.asarray(reference[:, 0]), np.asarray(hs))
    ]
    return jnp.stack(trajectories)


def model_rollout(model, reference: Field, hs: Field, dx: tuple[float, ...]) -> Field:
    """Trajectories batched while a batch stays below 8 x 256^2 cells."""
    n_traj, _, nx, ny, _ = reference.shape
    step = eqx.Partial(model, solver=llg_solver((nx, ny), dx))
    size = max(1, 8 * 256**2 // (nx * ny))
    chunks = [
        rollout_batch(step, reference[i : i + size, 0], hs[i : i + size], N_STEPS)  # type: ignore
        for i in range(0, n_traj, size)
    ]
    return jnp.concatenate(chunks)


def baseline(dataset: str, films: list[str], cells: list[float] | None):
    """magnum.np run directly on each coarse mesh: what the emulator has to beat."""
    # torch shares the GPU with JAX here, so JAX must not grab most of it up front
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for name, k in cases(films, cells):
        out = cache_dir(dataset) / f"baseline_{name}_k{k}.npz"
        if k == 1 or out.exists():  # at k = 1 magnum.np is the reference itself
            continue
        reference, hs, dx = load_truth(dataset, name, k)
        scores = save_metrics(out, magnum_rollout(reference, hs, dx), reference[:, 1:])
        print(f"baseline {name} k={k}: mse {scores['mse'].mean():.2e}", flush=True)


def rollouts(dataset: str, runs: list[str], films: list[str], cells: list[float] | None):
    """Each run's free rollout from the coarse-grained first frame of every trajectory."""
    for run in runs:
        model = load_model(run_dir(dataset, run))
        for name, k in cases(films, cells):
            out = run_dir(dataset, run) / "scaling" / f"model_{name}_k{k}.npz"
            if out.exists():
                continue
            reference, hs, dx = load_truth(dataset, name, k)
            scores = save_metrics(out, model_rollout(model, reference, hs, dx), reference[:, 1:])
            print(f"{run} {name} k={k}: mse {scores['mse'].mean():.2e}", flush=True)


def field_ramp(axis: int) -> Field:
    """(T, 3) applied field in A/m, one per step: +H_max -> -H_max -> +H_max along axis."""
    n_half = round(2 * H_MAX_MT / RAMP_MT_PER_NS / (DT * 1e9))
    down = jnp.linspace(H_MAX_MT, -H_MAX_MT, n_half, endpoint=False)
    up = jnp.linspace(-H_MAX_MT, H_MAX_MT, n_half, endpoint=False)
    ramp = jnp.concatenate([down, up]) * 1e-3 / MU_0
    return jnp.zeros((2 * n_half, 3)).at[:, axis].set(ramp)


def saturated_state(n: int, axis: int) -> Field:
    """Saturated along axis, relaxed at +H_max on the native mesh."""
    m = jnp.zeros((n, n, 3)).at[..., axis].set(1.0)
    relaxer = llg_solver((n, n), DX, alpha=1.0)
    return rollout(relaxer, m, field_ramp(axis)[0], RELAX_STEPS)[-1]


def sweep(step, m0: Field, h: Field) -> Field:
    """Bulk <m> (T, 3) along the field sequence h (T, 3)."""
    m, bulk = m0, []
    for i in range(0, len(h), SWEEP_CHUNK):
        frames = rollout(step, m, h[i : i + SWEEP_CHUNK], len(h[i : i + SWEEP_CHUNK]))  # type: ignore
        bulk.append(frames.mean(axis=(1, 2)))
        m = frames[-1]
    return jnp.concatenate(bulk)


def sweep_arm(path: Path, model, n: int, cells: list[float], starts: dict, ramps: dict):
    """Every loop of one arm (the coarse solver if model is None) missing from path."""
    results = dict(np.load(path)) if path.exists() else {}
    for axis, h in ramps.items():
        results[f"H_{axis}"] = h
        for nm in cells:
            if f"bulk_{axis}_{nm:g}" in results:
                continue
            k = round(nm / CELL_NM)
            solver = llg_solver((n // k, n // k), coarse_dx(k))
            step = solver if model is None else eqx.Partial(model, solver=solver)
            results[f"bulk_{axis}_{nm:g}"] = sweep(step, coarse_grain(starts[axis], k), h)  # type: ignore
            np.savez(path, **results)
            print(f"{path.stem}: {nm:g} nm, {axis} sweep done", flush=True)


def hysteresis(dataset: str, runs: list[str], films: list[str], cells: list[float] | None):
    """Quasi-static in-plane loops of square films: the 5 nm solver is the reference,
    the coarse solver and every run start from its relaxed state, area-averaged."""
    cells = sorted(set(cells or HYSTERESIS_CELLS) | {CELL_NM})
    for name in films:
        n = FILMS[name].cells
        out = RESULTS / dataset / ("hysteresis" if name == "sq256" else f"hysteresis_{name}")
        out.mkdir(parents=True, exist_ok=True)
        ramps = {axis: field_ramp(i) for axis, i in AXES.items()}
        starts = {axis: saturated_state(n, i) for axis, i in AXES.items()}
        sweep_arm(out / "solver.npz", None, n, cells, starts, ramps)
        for run in runs:
            model = load_model(run_dir(dataset, run))
            sweep_arm(out / f"{run.replace('/', '__')}.npz", model, n, cells, starts, ramps)
        plot_loops(out, runs, cells, f"{FILMS[name].side_um:g} um film, {RAMP_MT_PER_NS:g} mT/ns")


def plot_loops(out: Path, runs: list[str], cells: list[float], title: str):
    solver = dict(np.load(out / "solver.npz"))
    arms = {run: dict(np.load(out / f"{run.replace('/', '__')}.npz")) for run in runs}
    loops = {}
    print(f"{'axis':>4} {'cell':>5} {'arm':>24} {'Hc down':>8} {'Hc up':>7} {'area':>7}")
    for axis, i in AXES.items():
        h_mt = solver[f"H_{axis}"][:, i] * MU_0 * 1e3
        reference = solver[f"bulk_{axis}_{CELL_NM:g}"][:, i]
        loops[axis] = {}
        for nm in cells:
            row = {"reference": reference}
            if nm != CELL_NM:
                row["LLG solver"] = solver[f"bulk_{axis}_{nm:g}"][:, i]
            row |= {run: arm[f"bulk_{axis}_{nm:g}"][:, i] for run, arm in arms.items()}
            loops[axis][nm] = {arm: (h_mt, m) for arm, m in row.items()}
            for arm, m in row.items():
                if arm == "reference" and nm != CELL_NM:
                    continue
                hc_down, hc_up, area = (
                    metrics.loop_metrics(h_mt, m)[key] for key in ("hc_down", "hc_up", "area")
                )
                print(f"{axis:>4} {nm:>5g} {arm:>24} {hc_down:8.2f} {hc_up:7.2f} {area:7.1f}")
    tag = "_".join(run.replace("/", "-") for run in runs)
    plotting.save(plotting.hysteresis_loops(loops, title), out / f"loops_{tag}.png")


def load_curves(dataset: str, runs: list[str], film: str, k: int, metric: str) -> dict[str, Field]:
    """The solver's and each configuration's (seed-averaged) metric per rollout step."""
    curves = {}
    path = cache_dir(dataset) / f"baseline_{film}_k{k}.npz"
    if path.exists() and metric in np.load(path):
        curves["LLG solver"] = np.load(path)[metric]
    configurations = defaultdict(list)
    for run in runs:
        path = run_dir(dataset, run) / "scaling" / f"model_{film}_k{k}.npz"
        if path.exists() and metric in np.load(path):
            configurations[run.split("/")[0]].append(np.load(path)[metric])
    return curves | {
        name: jnp.mean(jnp.stack(seeds), axis=0) for name, seeds in configurations.items()
    }


def report(dataset: str, runs: list[str], films: list[str], cells: list[float] | None):
    """Per film: a table of mean rollout MSE against the coarse solver and a score (the
    geometric mean over cell sizes of the ratio, < 1 beats the solver), and figures."""
    out = cache_dir(dataset) / "report"
    summary, points = {}, {}
    for name in films:
        ks = factors(FILMS[name], cells)
        mse = {CELL_NM * k: load_curves(dataset, runs, name, k, "mse") for k in ks}
        summary[name] = score_table(name, mse)
        points[name] = defaultdict(dict)
        for nm, row in mse.items():
            for label, curve in row.items():
                points[name][label][nm] = float(curve.mean())
        for metric in ("mse", "vrmse"):
            curves = {CELL_NM * k: load_curves(dataset, runs, name, k, metric) for k in ks}
            title = f"{FILMS[name].side_um:g} um film ({name})"
            plotting.save(
                plotting.error_vs_step(curves, metric, title), out / f"{metric}_vs_step_{name}.png"
            )
    plotting.save(plotting.error_vs_cell(points, "mse"), out / "mse_vs_cell.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def score_table(film: str, mse: dict[float, dict[str, Field]]) -> dict:
    solver = {nm: row["LLG solver"].mean() for nm, row in mse.items() if "LLG solver" in row}
    labels = list(dict.fromkeys(label for row in mse.values() for label in row))
    print(f"\n{film}: mean rollout MSE x1e-3 (ratio to the LLG solver)")
    print(f"{'':>20} " + " ".join(f"{nm:>14g}" for nm in mse) + "   score")
    summary = {}
    for label in labels:
        means = {nm: float(row[label].mean()) for nm, row in mse.items() if label in row}
        ratios = {nm: means[nm] / solver[nm] for nm in means if nm in solver}
        score = (
            float(jnp.exp(jnp.log(jnp.array(list(ratios.values()))).mean()))
            if ratios
            else float("nan")
        )
        summary[label] = {"mse": means, "score": score}
        columns = [
            f"{1e3 * means[nm]:7.2f} ({ratios[nm]:4.2f})" if nm in ratios
            else f"{1e3 * means[nm]:7.2f}       " if nm in means
            else f"{'-':>14}"
            for nm in mse
        ]  # fmt: skip
        print(f"{label:>20} " + " ".join(columns) + f"   {score:6.3f}")
    return summary


def reference_frames(dataset: str, film: str, k: int, traj: int) -> tuple[Field, Field, tuple]:
    """One trajectory at factor k: from the truth cache, else coarse-grained from disk."""
    if (cache_dir(dataset) / f"truth_{film}_k{k}.npy").exists():
        reference, hs, dx = load_truth(dataset, film, k)
        return reference[traj], hs[traj], dx
    read, h = list(read_trajectories(FILMS[film].files(dataset)))[traj]
    frames = jnp.stack([coarse_grain(read(t), k) for t in range(N_STEPS + 1)])
    return frames.astype(jnp.float32), h.astype(jnp.float32), coarse_dx(k)  # type: ignore


def animate(dataset: str, runs: list[str], films: list[str], cells: list[float] | None, traj: int):
    """GIFs of the reference, each run and the coarse solver, and their cosine similarity
    to the reference per cell."""
    for run in runs:
        model = load_model(run_dir(dataset, run))
        out = run_dir(dataset, run) / "animations"
        out.mkdir(parents=True, exist_ok=True)
        for name in films:
            ks = [round(nm / CELL_NM) for nm in cells] if cells else FILMS[name].factors
            for k in ks:
                reference, h, dx = reference_frames(dataset, name, k, traj)
                solver = llg_solver(reference.shape[1:3], dx)
                arms = {run: eqx.Partial(model, solver=solver)}
                if k > 1:
                    arms["LLG solver"] = solver  # type: ignore
                tag = f"{name}_traj{traj}_{CELL_NM * k:g}nm"
                titles = [
                    f"{name}, {CELL_NM * k:g} nm cells, t = {t * DT * 1e9:.2f} ns"
                    for t in range(1, N_STEPS + 1)
                ]
                plotting.animate(
                    reference[1:], [t + ": reference" for t in titles], out / f"{tag}_reference.gif"
                )
                for arm, step in arms.items():
                    pred = rollout(step, reference[0], h, N_STEPS)  # type: ignore
                    label = arm.replace("/", "-").replace(" ", "-")
                    plotting.animate(
                        pred, [f"{t}: {arm}" for t in titles], out / f"{tag}_{label}.gif"
                    )
                    cos = (pred * reference[1:]).sum(axis=-1, keepdims=True)
                    plotting.animate(
                        cos, [f"{t}: {arm}" for t in titles], out / f"{tag}_{label}_cos.gif",
                        labels=("cosine similarity to the reference",), cmap="Blues_r", vmin=0.0,
                    )  # fmt: skip


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("validate", "truth", "baseline", "rollouts", "hysteresis", "report", "animate"),
    )
    parser.add_argument("--dataset", default="llg_field_switching")
    parser.add_argument("--runs", nargs="*", default=[], help="<configuration>/seed_<s>")
    parser.add_argument("--films", nargs="*", default=["valid256"], choices=list(FILMS))
    parser.add_argument("--cells", type=float, nargs="*", default=None, help="cell sizes in nm")
    parser.add_argument("--traj", type=int, default=4, help="the trajectory to animate")
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    if args.stage == "validate":
        validate(args.dataset, args.runs)
    elif args.stage == "truth":
        truth(args.dataset, args.films)
    elif args.stage == "baseline":
        baseline(args.dataset, args.films, args.cells)
    elif args.stage == "rollouts":
        rollouts(args.dataset, args.runs, args.films, args.cells)
    elif args.stage == "hysteresis":
        hysteresis(args.dataset, args.runs, args.films, args.cells)
    elif args.stage == "report":
        report(args.dataset, args.runs, args.films, args.cells)
    else:
        animate(args.dataset, args.runs, args.films, args.cells, args.traj)


if __name__ == "__main__":
    main()
