"""VRMSE (and MSE) against rollout step, one panel per cell size: the closure model against
the coarse LLG solver, from the scaling study's cache (`datagen/eval_scaling.py`).

    uv run python -m datagen.plot_vrmse_simple --film sq256 --cells 5 10 20 40 80 \
        --runs closure/seed_0 closure/seed_1

Writes `<out>/<metric>_simple_<film>_linear.png` and `_log.png` per `--metrics` (default `<out>` is the
shared report directory `results-v2/<dataset>/scaling/report/`). The runs' curves are
averaged, as the report does; the solver at 5 nm is the reference itself, so it has no
curve there.

By default the curves are the cache's means over all of the film's trajectories. With
`--trajs <indices>` they are means over those trajectories only, read from the
per-trajectory re-roll `datagen/pertraj.py` wrote (`pertraj_<film>.npz`); the file names
then carry the indices, e.g. `mse_simple_sq256_traj0-3_linear.png`. `--rank` prints every
trajectory's advantage (solver mean MSE over the run's, per cell size) to choose from.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from datagen.eval_scaling import cache_dir, run_dir
from datagen.scaling_report import CELL_NM, FILM_UM, _style

INK, SECOND = "#0b0b0b", "#52514e"
CLOSURE, SOLVER = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
# one hue per run in `--separate` mode: slots 1, 7, 3, 4, 5, 6, 8 (orange is the solver)
RUN_COLORS = ("#2a78d6", "#4a3aa7", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948")


def curves(cache, film, cells, runs, metric="vrmse", trajs=None):
    """cell nm -> (closure metric (T,) averaged over runs, the solver's (T,) or None);
    over all trajectories (the cache) or over `trajs` (the per-trajectory re-roll)."""
    out = {}
    per = dict(np.load(cache / f"pertraj_{film}.npz")) if trajs is not None else None
    for nm in cells:
        k = int(round(nm / CELL_NM))
        if per is None:
            model = [np.load(run_dir(cache, r) / f"model_{film}_k{k}.npz")[metric] for r in runs]
            base = cache / f"baseline_{film}_k{k}.npz"
            solver = np.load(base)[metric] if base.exists() else None
        else:
            model = [per[f"{r.replace('/', '__')}_{metric}_k{k}"][trajs].mean(0) for r in runs]
            key = f"solver_{metric}_k{k}"
            solver = per[key][trajs].mean(0) if key in per else None
        out[nm] = (np.mean(model, axis=0), solver)
    return out


def rank(cache, film, cells, runs):
    """Print each trajectory's advantage: the solver's mean MSE over the runs' mean, per
    cell size, and the geometric mean over the cell sizes that have a solver."""
    per = dict(np.load(cache / f"pertraj_{film}.npz"))
    ks = [int(round(nm / CELL_NM)) for nm in cells]
    ratios = []
    for k in ks:
        if f"solver_mse_k{k}" not in per:
            ratios.append(None)
            continue
        model = np.mean([per[f"{r.replace('/', '__')}_mse_k{k}"] for r in runs], axis=0)  # (traj, T)
        ratios.append(per[f"solver_mse_k{k}"].mean(1) / model.mean(1))
    have = [r for r in ratios if r is not None]
    geo = np.exp(np.mean(np.log(have), axis=0))
    print("solver mean MSE / closure mean MSE, per trajectory (rows) and cell size (columns)")
    print(f"{'traj':>4} " + " ".join(f"{nm:>6g}" for nm in cells) + "   geo-mean")
    for j in np.argsort(-geo):
        print(f"{j:>4} " + " ".join(f"{r[j]:6.2f}" if r is not None else "     -" for r in ratios)
              + f"   {geo[j]:6.2f}")  # fmt: skip


def plot(data, film, log, path, metric="vrmse", trajs=None):
    n = len(data)
    fig, axs = plt.subplots(1, n, figsize=(2.9 * n, 3.3), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    step = np.arange(1, len(next(iter(data.values()))[0]) + 1)
    for ax, (nm, (model, solver)) in zip(axs, data.items()):
        ax.plot(step, model, color=CLOSURE, lw=2.0, label="closure model")
        if solver is not None:
            ax.plot(step, solver, color=SOLVER, lw=2.0, label="LLG Solver")
        ax.set_title(f"{nm:g} nm cells" + (" (native)" if nm == CELL_NM else ""), fontsize=9, color=INK)
        ax.set_xlabel("rollout step (10 ps)", fontsize=8, color=SECOND)
        _style(ax)
    if log:
        axs[0].set_yscale("log")
    else:
        axs[0].set_ylim(bottom=0)
    axs[0].set_xlim(0, step[-1])
    axs[0].set_ylabel(metric.upper(), fontsize=9, color=SECOND)
    handles = [
        Line2D([], [], color=CLOSURE, lw=2.0, label="closure model"),
        Line2D([], [], color=SOLVER, lw=2.0, label="LLG Solver"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))  # fmt: skip
    fig.suptitle(f"{FILM_UM[film]:g} um film" + (f", trajectories {trajs}" if trajs is not None else ""),
                 fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_separate(by_run, film, log, path, metric, trajs, solver_cache):
    """One line per run instead of their mean: by_run = {run: curves(...)}."""
    cells = list(next(iter(by_run.values())))
    n = len(cells)
    fig, axs = plt.subplots(1, n, figsize=(2.9 * n, 3.6), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    step = np.arange(1, len(next(iter(by_run.values()))[cells[0]][0]) + 1)
    names = {r: ("previous closure" if r.split("/")[0] == "closure" else r.split("/")[0]) for r in by_run}
    print(f"{film} {metric}, mean over steps x1e-3 (ratio to the solver); trajectories {trajs}")
    print(f"{'run':>18} " + " ".join(f"{c:>13g}" for c in cells))
    line = f"{'LLG solver':>18} "
    for c in cells:
        solver = next(iter(by_run.values()))[c][1]
        line += f"{'reference':>13} " if solver is None else f"{1e3 * solver.mean():13.2f} "
    print(line)
    for ax, c in zip(axs, cells):
        solver = next(iter(by_run.values()))[c][1]
        if solver is not None:
            ax.plot(step, solver, color=SOLVER, lw=2.2, label="LLG Solver")
        for (run, data), color in zip(by_run.items(), RUN_COLORS):
            ax.plot(step, data[c][0], color=color, lw=1.6, label=names[run])
        ax.set_title(f"{c:g} nm cells" + (" (native)" if c == CELL_NM else ""), fontsize=9, color=INK)
        ax.set_xlabel("rollout step (10 ps)", fontsize=8, color=SECOND)
        _style(ax)
    for run, data in by_run.items():
        line = f"{names[run]:>18} "
        for c in cells:
            m, solver = data[c]
            line += f"{1e3 * m.mean():6.2f}" + (f" ({m.mean() / solver.mean():4.2f}) " if solver is not None else "        ")
        print(line)
    if log:
        axs[0].set_yscale("log")
    else:
        axs[0].set_ylim(bottom=0)
    axs[0].set_xlim(0, step[-1])
    axs[0].set_ylabel(metric.upper(), fontsize=9, color=SECOND)
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))  # fmt: skip
    fig.suptitle(f"{FILM_UM[film]:g} um film" + (f", trajectories {trajs}" if trajs is not None else ""),
                 fontsize=10, color=INK)  # fmt: skip
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--film", default="sq256")
    p.add_argument("--cells", type=float, nargs="+", default=(5, 10, 20, 40, 80), help="cell sizes in nm")
    p.add_argument("--runs", nargs="+", default=("closure/seed_0", "closure/seed_1"))
    p.add_argument("--metrics", nargs="+", default=("vrmse", "mse"), choices=("vrmse", "mse"))
    p.add_argument("--trajs", type=int, nargs="+", default=None, help="average over these trajectories only")
    p.add_argument("--rank", action="store_true", help="print each trajectory's advantage and exit")
    p.add_argument("--separate", action="store_true", help="one line per run instead of their mean")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cache = cache_dir(args.dataset)
    out = args.out or cache / "report"
    out.mkdir(parents=True, exist_ok=True)
    if args.rank:
        rank(cache, args.film, args.cells, args.runs)
        return
    tag = "" if args.trajs is None else "_traj" + "-".join(str(j) for j in args.trajs)
    # files are named by film and metric; runs other than the default closure pair get
    # their configuration names in the name too, so nothing overwrites the baseline's
    if tuple(args.runs) != ("closure/seed_0", "closure/seed_1"):
        tag = "_" + "-".join(sorted({r.split("/")[0] for r in args.runs})) + tag
    if max(args.cells) < 80:
        tag += f"_to{max(args.cells):g}nm"
    for metric in args.metrics:
        if args.separate:
            by_run = {r: curves(cache, args.film, args.cells, [r], metric, args.trajs) for r in args.runs}
            for log in (False, True):
                path = out / f"{metric}_compare_{args.film}{tag}_{'log' if log else 'linear'}.png"
                plot_separate(by_run, args.film, log, path, metric, args.trajs, cache)
                print(path)
            continue
        data = curves(cache, args.film, args.cells, args.runs, metric, args.trajs)
        for log in (False, True):
            path = out / f"{metric}_simple_{args.film}{tag}_{'log' if log else 'linear'}.png"
            plot(data, args.film, log, path, metric, args.trajs)
            print(path)


if __name__ == "__main__":
    main()
