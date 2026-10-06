"""Figures and summary for `datagen/eval_scaling.py`'s cached stages.

`report(cache, films, runs)` writes into `<cache>/report/`:

    summary.json              per film and cell size: rollout MSE (mean over the 1 ns
                              and at 1 ns) and bulk <m> MSE, per arm, naive coarse
                              micromagnetics and persistence, plus seconds per step
    error_vs_cell_size.png    rollout MSE against cell size, one panel per film
    error_vs_film_size.png    rollout MSE against film size, one panel per cell size
    rollout_large.png         MSE(t) on the 30.7 um film, one panel per cell size
    mse_vs_step.png           MSE against rollout step, films x cell sizes, each
                              configuration against micromagnetics on the same mesh
    vrmse_vs_step.png         the same for the well's VRMSE (recorded since 2026-10-02)
    bulk_large.png            <m_x>(t), <m_y>(t) on the 30.7 um film
    snapshots_large_k<k>.png  in-plane angle of m: truth, each arm, naive coarse
    runtime.png               seconds per 10 ps step against number of cells
    speed.png / speed.json    one 1 ns trajectory: the reference solver (5 nm) that
                              generated the data against the emulator, same mesh
    demo_bulk_<film>.png, demo_snapshots_<film>.png, demo.json
                              each demo film (`eval_scaling demo`), no truth

An *arm* is a configuration: its seeds are averaged (lines) and their spread shaded.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datagen.eval_scaling import FILMS, SNAPSHOTS, run_dir

L_EX_NM = 5.69  # sqrt(2A / (mu0 Ms^2)) for the dataset's permalloy
CELL_NM = 5.0
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e4e3df"
ARM_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#4a3aa7",
    "#e87ba4",
)  # categorical slots 1, 2, 7, 5
NAIVE_COLOR = "#1baf7a"  # slot 3
FILM_UM = {  # film side in um, 5 nm cells
    "sq64": 0.32,
    "sq128": 0.64,
    "sq256": 1.28,
    "sq512": 2.56,
    "sq1024": 5.12,
    "large": 30.72,
    "valid256": 1.28,
}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors="#52514e", labelsize=8)


def _log_ticks(ax, values, fmt="{:g}"):
    """Label a log x-axis at exactly the values plotted, without minor labels."""
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    values = sorted(set(values))
    ax.xaxis.set_major_locator(FixedLocator(values))
    ax.set_xticklabels([fmt.format(v) for v in values])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())


def _legend_below(fig, axs):
    """One legend for the figure, under the panels, from every panel's lines."""
    handles = {}
    for ax in np.ravel(axs):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center",
               ncol=min(4, len(handles)), frameon=False, fontsize=7)  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))


def _load(cache, films, runs):
    """-> {(film, k): {"naive": npz, "persistence": npz, arm: [npz per seed]}}"""
    arms = defaultdict(list)
    for run in runs:
        arms[run.split("/")[-2]].append(run)
    data = {}
    for film in films:
        for k in FILMS[film][1]:
            entry = {}
            for name, key in (("baseline", "naive"), ("persistence", "persistence")):
                p = cache / f"{name}_{film}_k{k}.npz"
                if p.exists():
                    entry[key] = _masked(np.load(p))
            for arm, arm_runs in arms.items():
                seeds = [
                    _masked(np.load(p))
                    for r in arm_runs
                    if (p := run_dir(cache, r) / f"model_{film}_k{k}.npz").exists()
                ]
                if seeds:
                    entry[arm] = seeds
            if entry:
                data[film, k] = entry
    return data, list(arms)


def _masked(npz):
    """A stage file as a dict, its MSE / VRMSE curves masked where NaN (a
    multi-frame model's context steps), so their means skip those steps."""
    d = dict(npz)
    for key in ("mse", "vrmse"):
        if key in d:
            d[key] = np.ma.masked_invalid(d[key])
    return d


def _bulk_mse(d):
    return float(np.nanmean((d["bulk_pred"] - d["bulk_ref"]) ** 2))


def _summary(data, arms):
    out = {}
    for (film, k), entry in data.items():
        row = {"cell_nm": CELL_NM * k, "film_um": FILM_UM[film]}
        for name in ("naive", "persistence"):
            if name in entry:
                d = entry[name]
                row[name] = {
                    "mse_mean": float(d["mse"].mean()),
                    "mse_final": float(d["mse"][-1]),
                    "bulk_mse": _bulk_mse(d),
                    "seconds_per_step": float(d["seconds_per_step"]),
                }
        if "naive" not in entry and k == 1:
            row["naive"] = "the reference solver itself (exact)"
        for arm in arms:
            if arm in entry:
                seeds = entry[arm]
                row[arm] = {
                    "mse_mean": [float(d["mse"].mean()) for d in seeds],
                    "mse_final": [float(d["mse"][-1]) for d in seeds],
                    "bulk_mse": [_bulk_mse(d) for d in seeds],
                    "seconds_per_step": float(
                        np.median([d["seconds_per_step"] for d in seeds])
                    ),
                }
        out[f"{film}@{CELL_NM * k:g}nm"] = row
    return out


def _arm_band(ax, x, seeds_y, color, label):
    y = np.asarray(seeds_y)
    ax.plot(x, y.mean(axis=0), color=color, lw=1.8, label=label, marker="o", ms=4)
    if len(y) > 1:
        ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=color, alpha=0.15, lw=0)


def _error_vs_cell(data, arms, films, path):
    films = [f for f in films if any((f, k) in data for k in FILMS[f][1])]
    fig, axs = plt.subplots(1, len(films), figsize=(3.2 * len(films), 3.0), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, film in zip(axs, films):
        ks = [k for k in FILMS[film][1] if (film, k) in data]
        cells = [CELL_NM * k for k in ks]
        for i, arm in enumerate(arms):
            kk = [k for k in ks if arm in data[film, k]]
            if kk:
                n = min(len(data[film, k][arm]) for k in kk)
                ys = [
                    [data[film, k][arm][s]["mse"].mean() for k in kk] for s in range(n)
                ]
                _arm_band(ax, [CELL_NM * k for k in kk], ys, ARM_COLORS[i], arm)
        kk = [k for k in ks if "naive" in data[film, k]]
        ax.plot(
            [CELL_NM * k for k in kk],
            [data[film, k]["naive"]["mse"].mean() for k in kk],
            color=NAIVE_COLOR, lw=1.8, marker="s", ms=4, label="coarse solver (magnum.np at that cell)",
        )  # fmt: skip
        ax.plot(
            cells,
            [data[film, k]["persistence"]["mse"].mean() for k in ks],
            color=MUTED, lw=1.2, ls="--", label="persistence m(t) = m(0)",
        )  # fmt: skip
        ax.axvline(L_EX_NM, color=MUTED, lw=0.8, ls=":")
        ax.text(L_EX_NM * 1.05, 0.98, "l_ex", transform=ax.get_xaxis_transform(),
                fontsize=7, color="#52514e", va="top")  # fmt: skip
        ax.set(xscale="log", yscale="log", title=f"{film} ({FILM_UM[film]:g} um)")
        _log_ticks(ax, cells)
        ax.set_xlabel("cell size (nm)", fontsize=8)
        ax.title.set_fontsize(9)
        _style(ax)
    axs[0].set_ylabel("rollout MSE, mean over 1 ns", fontsize=8)
    _legend_below(fig, axs)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _error_vs_film(data, arms, films, path, ks=(4, 8, 16)):
    fig, axs = plt.subplots(1, len(ks), figsize=(3.3 * len(ks), 3.0), sharey=True)
    for ax, k in zip(axs, ks):
        fs = [f for f in films if (f, k) in data]
        x = [FILM_UM[f] for f in fs]
        for i, arm in enumerate(arms):
            ff = [f for f in fs if arm in data[f, k]]
            if ff:
                n = min(len(data[f, k][arm]) for f in ff)
                ys = [[data[f, k][arm][s]["mse"].mean() for f in ff] for s in range(n)]
                _arm_band(ax, [FILM_UM[f] for f in ff], ys, ARM_COLORS[i], arm)
        ff = [f for f in fs if "naive" in data[f, k]]
        ax.plot([FILM_UM[f] for f in ff], [data[f, k]["naive"]["mse"].mean() for f in ff],
                color=NAIVE_COLOR, lw=1.8, marker="s", ms=4,
                label="coarse solver (magnum.np at that cell)")  # fmt: skip
        ax.plot(x, [data[f, k]["persistence"]["mse"].mean() for f in fs], color=MUTED,
                lw=1.2, ls="--", label="persistence")  # fmt: skip
        ax.axvline(1.28, color=MUTED, lw=0.8, ls=":")
        ax.text(1.28 * 1.05, 0.98, "training film", transform=ax.get_xaxis_transform(),
                fontsize=7, color="#52514e", va="top")  # fmt: skip
        ax.set(xscale="log", yscale="log", title=f"{CELL_NM * k:g} nm cells")
        _log_ticks(ax, x)
        ax.set_xlabel("film side (um)", fontsize=8)
        ax.title.set_fontsize(9)
        _style(ax)
    axs[0].set_ylabel("rollout MSE, mean over 1 ns", fontsize=8)
    _legend_below(fig, axs)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _rollout_large(data, arms, path):
    ks = [k for k in FILMS["large"][1] if ("large", k) in data]
    fig, axs = plt.subplots(1, len(ks), figsize=(3.0 * len(ks), 2.9), sharey=True)
    axs = np.atleast_1d(axs)
    t = np.arange(1, 101) * 0.01  # ns
    for ax, k in zip(axs, ks):
        e = data["large", k]
        for i, arm in enumerate(arms):
            if arm in e:
                y = np.stack([d["mse"] for d in e[arm]])
                ax.plot(t, y.mean(0), color=ARM_COLORS[i], lw=1.8, label=arm)
                if len(y) > 1:
                    ax.fill_between(t, y.min(0), y.max(0), color=ARM_COLORS[i],
                                    alpha=0.15, lw=0)  # fmt: skip
        if "naive" in e:
            ax.plot(t, e["naive"]["mse"], color=NAIVE_COLOR, lw=1.8,
                    label="coarse solver (magnum.np at that cell)")  # fmt: skip
        ax.plot(t, e["persistence"]["mse"], color=MUTED, lw=1.2, ls="--",
                label="persistence")  # fmt: skip
        ax.set(yscale="log", title=f"{CELL_NM * k:g} nm ({6144 // k}^2 cells)")
        ax.set_xlabel("time (ns)", fontsize=8)
        ax.title.set_fontsize(9)
        _style(ax)
    axs[0].set_ylabel("MSE vs magnum.np data (coarse-grained)", fontsize=8)
    fig.suptitle("30.7 um film: free rollout from the coarse-grained first frame",
                 fontsize=9)  # fmt: skip
    _legend_below(fig, axs)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_vs_step(
    data,
    arms,
    path,
    metric="mse",
    films=("sq256", "sq1024", "large"),
    cells=(10, 20, 40, 80),
):
    """Rollout `metric` ("mse" or "vrmse") against step, each configuration against
    micromagnetics on the same coarse mesh (magnum.np), one row per film and one column
    per cell size; entries cached before the metric was recorded are left out. Colours
    follow each configuration's index in `arms`, as in the other figures."""
    name = {"mse": "MSE", "vrmse": "VRMSE"}[metric]
    fig, axs = plt.subplots(len(films), len(cells), figsize=(3.0 * len(cells), 2.5 * len(films)),
                            sharex=True, squeeze=False)  # fmt: skip
    step = np.arange(1, 101)
    for r, film in enumerate(films):
        for c, cell in enumerate(cells):
            ax = axs[r, c]
            e = data.get((film, round(cell / CELL_NM)))
            if not e:
                ax.set_axis_off()
                continue
            for i, arm in enumerate(arms):
                if arm in e and all(metric in d for d in e[arm]):
                    y = np.stack([d[metric] for d in e[arm]])
                    ax.plot(step, y.mean(0), color=ARM_COLORS[i], lw=1.6, label=arm)
                    if len(y) > 1:
                        ax.fill_between(step, y.min(0), y.max(0), color=ARM_COLORS[i],
                                        alpha=0.15, lw=0)  # fmt: skip
            if metric in e.get("naive", {}):
                ax.plot(step, e["naive"][metric], color=NAIVE_COLOR, lw=1.8,
                        label="coarse solver (magnum.np at that cell)")  # fmt: skip
            ax.set_yscale("log")
            if r == 0:
                ax.set_title(f"{cell} nm cells", fontsize=9)
            if all(not a.axison for a in axs[r, :c]):
                ax.set_ylabel(
                    f"{film} ({FILM_UM[film]:g} um)\n{name} vs magnum.np data",
                    fontsize=8,
                )
            if r == len(films) - 1:
                ax.set_xlabel("rollout step (10 ps)", fontsize=8)
            _style(ax)
    _legend_below(fig, axs)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _bulk_large(data, arms, path):
    ks = [k for k in FILMS["large"][1] if ("large", k) in data]
    fig, axs = plt.subplots(2, len(ks), figsize=(3.0 * len(ks), 4.4), sharex=True,
                            sharey="row")  # fmt: skip
    axs = np.atleast_2d(axs).reshape(2, -1)
    t = np.arange(1, 101) * 0.01
    for j, k in enumerate(ks):
        e = data["large", k]
        ref = e["persistence"]["bulk_ref"][0]
        for c, name in enumerate(("<m_x>", "<m_y>")):
            ax = axs[c, j]
            ax.plot(t, ref[:, c], color=INK, lw=2.2, label="truth (5 nm, averaged)")
            for i, arm in enumerate(arms):
                if arm in e:
                    ax.plot(t, e[arm][0]["bulk_pred"][0][:, c], color=ARM_COLORS[i],
                            lw=1.6, label=arm)  # fmt: skip
            if "naive" in e:
                ax.plot(t, e["naive"]["bulk_pred"][0][:, c], color=NAIVE_COLOR, lw=1.6,
                        ls="-.", label="coarse solver (magnum.np at that cell)")  # fmt: skip
            if j == 0:
                ax.set_ylabel(name, fontsize=8)
            if c == 0:
                ax.set_title(f"{CELL_NM * k:g} nm", fontsize=9)
            else:
                ax.set_xlabel("time (ns)", fontsize=8)
            _style(ax)
    _legend_below(fig, axs[0])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _snapshots_large(data, arms, out):
    for k in FILMS["large"][1]:
        e = data.get(("large", k))
        if not e:
            continue
        rows = [("truth", e["persistence"]["snap_ref"])]
        for arm in arms:
            if arm in e:
                best = min(e[arm], key=lambda d: d["mse"].mean())
                rows.append((arm, best["snap_pred"]))
        if "naive" in e:
            rows.append(("micromagnetics\nat that cell", e["naive"]["snap_pred"]))
        fig, axs = plt.subplots(len(rows), len(SNAPSHOTS),
                                figsize=(2.1 * len(SNAPSHOTS), 2.1 * len(rows)))  # fmt: skip
        for r, (name, snaps) in enumerate(rows):
            for c, step in enumerate(SNAPSHOTS):
                ax = axs[r, c]
                snap = snaps[c].astype(np.float32)
                angle = np.arctan2(snap[..., 1], snap[..., 0]).T
                ax.imshow(angle, cmap="twilight", vmin=-np.pi, vmax=np.pi,
                          origin="lower", interpolation="nearest")  # fmt: skip
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"t = {(step + 1) * 0.01:.2f} ns", fontsize=8)
                if c == 0:
                    ax.set_ylabel(name, fontsize=8)
        fig.suptitle(f"30.7 um film at {CELL_NM * k:g} nm cells: in-plane angle of m",
                     fontsize=9)  # fmt: skip
        fig.tight_layout()
        fig.savefig(out / f"snapshots_large_k{k}.png", dpi=150)
        plt.close(fig)


def _runtime(data, arms, path):
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    emu, naive = {}, {}
    for (film, k), e in data.items():
        n = (FILMS_CELLS[film] // k) ** 2
        for arm in arms:
            if arm in e:
                emu.setdefault(n, []).append(float(e[arm][0]["seconds_per_step"]))
        if "naive" in e:
            naive.setdefault(n, []).append(float(e["naive"]["seconds_per_step"]))
    for series, color, label, marker in (
        (emu, ARM_COLORS[0], "emulator (one 10 ps step)", "o"),
        (naive, NAIVE_COLOR, "micromagnetics (one 10 ps step)", "s"),
    ):
        n = sorted(series)
        ax.plot(n, [np.median(series[i]) for i in n], color=color, lw=1.8,
                marker=marker, ms=4, label=label)  # fmt: skip
    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("cells in the mesh", fontsize=8)
    ax.set_ylabel("seconds per step (one MIG slice)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _speed(cache, runs, path):
    """Time for one 1 ns trajectory at 5 nm: the reference solver that generated the
    data against each configuration on the same mesh (its first seed's sweep)."""
    solver = json.loads((cache / "speed_solver.json").read_text())
    arms = {}
    for run in runs:
        f = run_dir(cache, run) / "speed_model.json"
        if f.exists():
            arms.setdefault(run.split("/")[-2], json.loads(f.read_text()))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    rows = sorted(solver.values(), key=lambda r: r["cells"])
    slice_rows = [r for r in rows if "MIG" in r["device"]]
    ax.plot([r["cells"] for r in slice_rows],
            [100 * r["seconds_per_step"] for r in slice_rows], color=INK, lw=1.8,
            marker="s", ms=4, label="reference solver, 5 nm (magnum.np)")  # fmt: skip
    full = [r for r in rows if "MIG" not in r["device"]]
    if full:
        ax.plot([r["cells"] for r in full], [100 * r["seconds_per_step"] for r in full],
                color=INK, ls="none", marker="*", ms=9,
                label="reference solver, 6144^2 on a full GPU (recorded)")  # fmt: skip
    for i, (arm, emu) in enumerate(arms.items()):
        e = sorted(emu.values(), key=lambda r: r["cells"])
        ax.plot([r["cells"] for r in e], [100 * r["seconds_per_step"] for r in e],
                color=ARM_COLORS[i], lw=1.8, marker="o", ms=4,
                label=f"{arm}, same mesh")  # fmt: skip
    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("cells in the mesh", fontsize=8)
    ax.set_ylabel("seconds per 1 ns trajectory (100 steps)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    speedup = {}
    for arm, emu in arms.items():
        speedup[arm] = {
            film: row["seconds_per_step"] / emu[side]["seconds_per_step"]
            for film, row in solver.items()
            if (side := str(round(np.sqrt(row["cells"])))) in emu
        }
    return {"solver_5nm": solver, "models_5nm": arms, "speedup_same_mesh": speedup}


def _demo(demo_file, cache, path_bulk, path_snap):
    """One demo film (`eval_scaling demo`): <m>(t) of coarse micromagnetics and each
    run, with the 30.7 um film's truth (same material and field) as the physical
    reference, and in-plane angle snapshots."""
    d = dict(np.load(demo_file))
    runs = sorted(k[: -len("_bulk")] for k in d if k.endswith("_bulk"))
    runs = ["solver"] + [r for r in runs if r != "solver"]
    ref = np.load(cache / "persistence_large_k16.npz")["bulk_ref"][0]
    t = np.arange(1, 101) * 0.01
    colors = {"solver": NAIVE_COLOR}
    for i, r in enumerate(r for r in runs if r != "solver"):
        colors[r] = ARM_COLORS[i]
    label = {r: f"{r.split('__')[-2]}, {r.split('__')[-1].replace('_', ' ')}"
             if "__" in r else "coarse solver (magnum.np at that cell)" for r in runs}  # fmt: skip
    side_mm = float(d["cells"]) * float(d["cell_nm"]) * 1e-6
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0), sharex=True)
    for c, (ax, name) in enumerate(zip(axs, ("<m_x>", "<m_y>"))):
        ax.plot(t, ref[:, c], color=INK, lw=2.2, label="30.7 um film truth (5 nm)")
        for r in runs:
            ax.plot(t, d[f"{r}_inner"][:, c], color=colors[r], lw=1.6,
                    ls="-." if r == "solver" else "-", label=label[r])  # fmt: skip
        ax.set_ylabel(f"{name}, interior", fontsize=8)
        ax.set_xlabel("time (ns)", fontsize=8)
        _style(ax)
    axs[-1].legend(frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"{side_mm:.2g} mm film at {float(d['cell_nm']):.0f} nm cells "
                 f"({int(d['cells'])}^2): interior magnetisation", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(path_bulk, dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(len(runs), len(SNAPSHOTS),
                            figsize=(2.1 * len(SNAPSHOTS), 2.1 * len(runs)))  # fmt: skip
    axs = np.atleast_2d(axs)
    for r_i, r in enumerate(runs):
        for c, step in enumerate(SNAPSHOTS):
            snap = d[f"{r}_snap"][c].astype(np.float32)
            ax = axs[r_i, c]
            ax.imshow(np.arctan2(snap[1], snap[0]).T, cmap="twilight", vmin=-np.pi,
                      vmax=np.pi, origin="lower", interpolation="nearest")  # fmt: skip
            ax.set_xticks([])
            ax.set_yticks([])
            if r_i == 0:
                ax.set_title(f"t = {(step + 1) * 0.01:.2f} ns", fontsize=8)
            if c == 0:
                ax.set_ylabel(label[r].replace(" at that cell", "\nat that cell"),
                              fontsize=8)  # fmt: skip
    fig.suptitle(f"{side_mm:.2g} mm film: in-plane angle of m", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_snap, dpi=150)
    plt.close(fig)
    return {
        "cells": int(d["cells"]),
        "cell_nm": float(d["cell_nm"]),
        "seconds_per_step": {r: float(d[f"{r}_seconds_per_step"]) for r in runs},
        "interior_m_final": {r: d[f"{r}_inner"][-1].tolist() for r in runs},
        "reference_m_final_30um": ref[-1].tolist(),
    }


FILMS_CELLS = {"sq64": 64, "sq128": 128, "sq256": 256, "sq512": 512, "sq1024": 1024,
               "large": 6144}  # fmt: skip


def report(cache, films, runs, out=None):
    if out is None:
        out = run_dir(cache, runs[0]) / "report" if len(runs) == 1 else cache / "report"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data, arms = _load(Path(cache), films, runs)
    (out / "summary.json").write_text(json.dumps(_summary(data, arms), indent=1))
    # the three most informative films, or whichever were run
    cell_films = [f for f in ("sq256", "sq1024", "large") if f in films] or films
    _error_vs_cell(data, arms, cell_films, out / "error_vs_cell_size.png")
    _error_vs_film(data, arms, films, out / "error_vs_film_size.png")
    for metric in ("mse", "vrmse"):
        _metric_vs_step(
            data, arms, out / f"{metric}_vs_step.png", metric, films=cell_films
        )
    if any(f == "large" for f, _ in data):
        _rollout_large(data, arms, out / "rollout_large.png")
        _bulk_large(data, arms, out / "bulk_large.png")
        _snapshots_large(data, arms, out)
    _runtime(data, arms, out / "runtime.png")
    if (Path(cache) / "speed_solver.json").exists() and runs:
        speed = _speed(Path(cache), runs, out / "speed.png")
        (out / "speed.json").write_text(json.dumps(speed, indent=1))
    demos = {}
    demo_files = [f for r in runs for f in sorted(run_dir(cache, r).glob("demo_*.npz"))]
    for f in demo_files:
        # <configuration>__seed_<s>__<side>um_<cell>nm
        tag = (
            f.parent.parent.relative_to(Path(cache).parent)
            .as_posix()
            .replace("/", "__")
        )
        tag += "__" + f.stem.removeprefix("demo_")
        demos[tag] = _demo(
            f,
            Path(cache),
            out / f"demo_bulk_{tag}.png",
            out / f"demo_snapshots_{tag}.png",
        )
    if demos:
        (out / "demo.json").write_text(json.dumps(demos, indent=1))
    print(f"wrote {out}")
