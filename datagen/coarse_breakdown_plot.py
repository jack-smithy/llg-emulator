"""Figures and summary for `datagen/coarse_breakdown.py`'s sweep.

`plot(out_dir, film)` reads `<out_dir>/<film>.npz` + `.json` and writes next to them:

    <film>_summary.json         per cell size (all trajectories and per initial state):
                                MSE, median / 90th-pct angular error and the fraction of
                                cells > 30 deg off at 1 ns and at the end, persistence and
                                one-step MSE, bulk <m> RMS error (0-1 ns, 0-end) and final
                                error, energy ratios to the 5 nm reference, seconds per step
    <film>_pointwise_vs_cell.png   MSE, median angle (90th pct dotted), fraction off, at
                                   1 ns (top) and at the end (bottom), per initial state
    <film>_error_vs_time.png       MSE(t) and median angle(t), cell sizes as lines, one
                                   column per initial state
    <film>_bulk_m.png              <m_x>(t), <m_y>(t) of every trajectory: reference and
                                   coarse solvers
    <film>_bulk_vs_cell.png        bulk <m> errors and energy ratios against cell size
    <film>_where_and_onestep.png   textured vs smooth cells, textured fraction, one-step
                                   against rollout error
    <film>_spectra.png             error power / reference power per wavelength at 1 ns
    <film>_snapshots.png           in-plane angle maps: reference and coarse solvers
    <film>_settle.png              time for <m>(t) to settle, per trajectory, against cell size
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from datagen.coarse_breakdown import ANGLE_DEG, IC_TYPES, L_EX_NM, METRICS, TEXTURE_NORM

INK, MUTED, GRID, SECOND = "#0b0b0b", "#8a8984", "#e4e3df", "#52514e"
IC_COLORS = dict(zip(IC_TYPES, ("#2a78d6", "#eb6834", "#1baf7a")))  # slots 1, 2, 3
RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")  # blue 250..700
SHOW_NM = (10, 20, 40, 80, 160)  # cell sizes drawn as lines where not all fit
MU_0 = 4e-7 * np.pi


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=SECOND, labelsize=8)


def _cell_axis(ax, cells, top=True):
    """Log x-axis in nm labelled at the cell sizes swept, l_ex on top."""
    ax.set_xscale("log")
    ticks = [c for c in sorted(cells) if c in (5, 10, 20, 40, 80, 160) or c < 8]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:.3g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("cell size (nm)", fontsize=8, color=SECOND)
    ax.axvline(L_EX_NM, color=MUTED, lw=0.8, ls=":")
    if top:
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / L_EX_NM, lambda x: x * L_EX_NM))
        sec.set_xticks([1, 2, 4, 8, 16, 28])
        sec.set_xticklabels(["1", "2", "4", "8", "16", "28"])
        sec.xaxis.set_minor_locator(NullLocator())
        sec.tick_params(colors=SECOND, labelsize=7)
        sec.spines["top"].set_color(MUTED)
        sec.set_xlabel("cell / l_ex", fontsize=7, color=SECOND)


def _legend_below(fig, axs, ncol=4):
    handles = {}
    for ax in np.ravel(axs):
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=ncol,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))  # fmt: skip


def _groups(ic):
    """name -> trajectory indices: every initial state present, then all."""
    g = {t: [j for j, x in enumerate(ic) if x == t] for t in IC_TYPES}
    g = {k: v for k, v in g.items() if v}
    g["all"] = list(range(len(ic)))
    return g


def _nearest(cells, nm):
    return int(np.argmin(np.abs(np.asarray(cells) - nm)))


def _shown(cells, want=SHOW_NM, k=5):
    """(mesh index, cell nm) of the cell sizes drawn as lines: `want` where swept, else
    (a partial sweep) the coarsest `k` meshes after the control."""
    idx = []
    for nm in want:
        i = _nearest(cells, nm)
        if abs(cells[i] - nm) < 0.5 and i not in idx:
            idx.append(i)
    if len(idx) < 2:
        cand = [i for i in range(len(cells)) if cells[i] > 5.5] or list(range(len(cells)))
        idx = cand[-k:]
    return [(i, float(cells[i])) for i in idx]


def _nanmean(x):
    x = np.asarray(x, float)
    return float(x[~np.isnan(x)].mean()) if (~np.isnan(x)).any() else float("nan")


def load(out_dir, film):
    d = dict(np.load(out_dir / f"{film}.npz"))
    meta = json.loads((out_dir / f"{film}.json").read_text())
    n = meta["n_done"]
    for k, v in d.items():  # a killed job's file holds n_done valid meshes
        if v.ndim and v.shape[0] == len(meta["cell_nm"]) and k not in ("H", "bulk_truth", "E_truth_fine"):
            d[k] = v[:n]
    d["cell_nm"] = d["cell_nm"][:n]
    d["ns"] = d["ns"][:n]
    return d, meta


# --- figures ----------------------------------------------------------------


def closure_floor(out_dir):
    """`datagen/predictability.py`'s closure floor at 1 ns, cell nm -> MSE, if it ran:
    the reference solver restarted with sub-cell noise, scored on that mesh."""
    f = Path(out_dir).parent / "predictability" / "mse.npz"
    if not f.exists():
        return {}
    z = np.load(f)
    return {5.0 * k: float(z[f"subcell_k{k}_k{k}"][99]) for k in (2, 4, 8) if f"subcell_k{k}_k{k}" in z.files}


def fig_pointwise(d, meta, groups, s1, s_end, path, floor=None):
    cells = d["cell_nm"]
    M = d["metrics"]
    im = {m: i for i, m in enumerate(METRICS)}
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    t_end = d["steps_cmp"][s_end] * meta["dt"] * 1e9
    for row, (s, when) in enumerate(((s1, "1 ns"), (s_end, f"{t_end:g} ns"))):
        for name, idx in groups.items():
            c = INK if name == "all" else IC_COLORS[name]
            lw = 2.2 if name == "all" else 1.6
            kw = dict(color=c, lw=lw, marker="o", ms=4)
            axs[row, 0].plot(cells, M[:, idx, s, im["mse"]].mean(1), label=name, **kw)
            axs[row, 1].plot(cells, M[:, idx, s, im["ang50"]].mean(1), label=name, **kw)
            axs[row, 1].plot(cells, M[:, idx, s, im["ang90"]].mean(1), color=c, lw=1.0,
                             ls=":", label="90th percentile (dotted)" if name == "all" else None)  # fmt: skip
            axs[row, 2].plot(cells, M[:, idx, s, im["frac_off"]].mean(1), label=name, **kw)
        axs[row, 0].plot(cells, d["persist_mse"][:, :, s].mean(1), color=MUTED, lw=1.2,
                         ls="--", label="persistence m(t) = m(0)")  # fmt: skip
        if row == 0 and floor:
            axs[0, 0].plot(list(floor), list(floor.values()), color=SECOND, ls="none", marker="D", ms=5,
                           label="closure floor: 5 nm solver restarted with sub-cell noise")  # fmt: skip
        axs[row, 0].set_yscale("log")
        axs[row, 0].set_ylabel(f"MSE at {when}", fontsize=9)
        axs[row, 1].set_ylabel(f"angular error at {when} (deg): median", fontsize=9)
        axs[row, 2].set_ylabel(f"cells > {ANGLE_DEG:g} deg off at {when}", fontsize=9)
        axs[row, 1].set_yscale("log")
        axs[row, 2].set_ylim(0, 1)
    for ax in axs.ravel():
        _style(ax)
        _cell_axis(ax, cells, top=ax in axs[0])
    fig.suptitle(f"{film_title(meta)}: coarse micromagnetics against the 5 nm reference, pointwise", fontsize=11)
    _legend_below(fig, axs, ncol=6)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_error_vs_time(d, meta, groups, path):
    cells = d["cell_nm"]
    t = d["steps_cmp"] * meta["dt"] * 1e9
    M = d["metrics"]
    im = {m: i for i, m in enumerate(METRICS)}
    shown = _shown(cells)
    names = list(groups)
    fig, axs = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.4), sharex=True, sharey="row")
    for col, name in enumerate(names):
        idx = groups[name]
        for ax, metric in zip(axs[:, col], ("mse", "ang50")):
            if cells[0] <= 5.5:
                ax.plot(t, M[0, idx, :, im[metric]].mean(0), color=MUTED, lw=1.2,
                        label=f"control: 5 nm rerun (the floor)")  # fmt: skip
            for (i, nm), c in zip(shown, RAMP):
                ax.plot(t, M[i, idx, :, im[metric]].mean(0), color=c, lw=1.8, label=f"{nm:.3g} nm cells")
            if metric == "mse":
                ax.plot(t, d["persist_mse"][shown[0][0], idx].mean(0), color=MUTED, lw=1.2,
                        ls="--", label="persistence")  # fmt: skip
            ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _style(ax)
        axs[0, col].set_title(f"{name} ({len(idx)} traj.)", fontsize=9)
        axs[1, col].set_xlabel("time (ns)", fontsize=8, color=SECOND)
    axs[0, 0].set_ylabel("MSE", fontsize=9)
    axs[1, 0].set_ylabel("median angular error (deg)", fontsize=9)
    fig.suptitle(f"{film_title(meta)}: error growth; data ends at 1 ns (dotted), reference continued by magnum.np at 5 nm", fontsize=10)
    _legend_below(fig, axs, ncol=7)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bulk_m(d, meta, path):
    cells = d["cell_nm"]
    n_traj = meta["n_traj"]
    t = np.arange(meta["n_steps"] + 1) * meta["dt"] * 1e9
    shown = _shown(cells, (10, 20, 40, 80), 4)
    ncol = 4
    nrow = int(np.ceil(n_traj / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), sharex=True, sharey=True)
    for j, ax in enumerate(axs.ravel()):
        if j >= n_traj:
            ax.set_visible(False)
            continue
        for comp, ls in ((0, "-"), (1, "--")):
            ax.plot(t, d["bulk_truth"][j, :, comp], color=INK, lw=2.0, ls=ls,
                    label=f"reference 5 nm: <m_{'xy'[comp]}>" )  # fmt: skip
            for (i, nm), c in zip(shown, RAMP[1:]):
                ax.plot(t, d["bulk_pred"][i, j, :, comp], color=c, lw=1.4, ls=ls,
                        label=f"{nm:.3g} nm cells: <m_{'xy'[comp]}>")  # fmt: skip
        h_mt = np.asarray(d["H"][j]) * MU_0 * 1e3
        ax.set_title(f"traj. {j}: {meta['ic'][j]}, H = ({h_mt[0]:.0f}, {h_mt[1]:.0f}) mT", fontsize=8)
        ax.axvline(1.0, color=MUTED, lw=0.8, ls=":")
        _style(ax)
        if j // ncol == nrow - 1:
            ax.set_xlabel("time (ns)", fontsize=8, color=SECOND)
    axs[0, 0].set_ylabel("<m>", fontsize=9)
    if nrow > 1:
        axs[1, 0].set_ylabel("<m>", fontsize=9)
    fig.suptitle(f"{film_title(meta)}: bulk magnetisation, reference against coarse micromagnetics", fontsize=10)
    _legend_below(fig, axs, ncol=5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def bulk_errors(d, meta):
    """(n_mesh, n_traj) RMS over 0-1 ns and 0-end of |<m>_coarse - <m>_ref|, and the final |.|."""
    diff = np.linalg.norm(d["bulk_pred"] - d["bulk_truth"][None], axis=-1)  # (mesh, traj, t)
    n1 = int(round(1e-9 / meta["dt"]))
    rms1 = np.sqrt((diff[:, :, : n1 + 1] ** 2).mean(-1))
    rms = np.sqrt((diff**2).mean(-1))
    return rms1, rms, diff[:, :, -1]


SETTLE = 0.02  # |<m>(t) - <m>(end)| below this from t on = settled


def settle_times(bulk, dt):
    """(...,) ns at which each <m>(t) trajectory ((..., T, 3)) has settled: the first
    time after which it stays within `SETTLE` of its final value."""
    dev = np.linalg.norm(bulk - bulk[..., -1:, :], axis=-1)  # (..., T)
    # the last step that is still outside, +1, scanning from the end
    outside = dev > SETTLE
    last = outside.shape[-1] - 1 - np.argmax(outside[..., ::-1], axis=-1)
    last = np.where(outside.any(axis=-1), last + 1, 0)
    return last * dt * 1e9


def fig_settle(d, meta, path):
    cells = d["cell_nm"]
    ic = meta["ic"]
    t_pred = settle_times(d["bulk_pred"], meta["dt"])  # (mesh, traj)
    t_ref = settle_times(d["bulk_truth"], meta["dt"])  # (traj,)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for j in range(meta["n_traj"]):
        c = IC_COLORS[ic[j]]
        axs[0].plot(cells, t_pred[:, j], color=c, lw=1.4, marker="o", ms=3, label=ic[j])
        axs[0].plot([cells[0]], [t_ref[j]], color=c, marker="D", ms=6, ls="none",
                    label="reference (diamond)" if j == 0 else None)  # fmt: skip
        axs[1].plot(cells, t_pred[:, j] / t_ref[j], color=c, lw=1.4, marker="o", ms=3, label=ic[j])
    axs[1].axhline(1.0, color=MUTED, lw=0.8)
    axs[0].set_ylabel(f"time to settle within {SETTLE:g} of the final <m> (ns)", fontsize=9)
    axs[1].set_ylabel("settling time / reference's", fontsize=9)
    axs[1].set_yscale("log")
    for ax in axs:
        _style(ax)
        _cell_axis(ax, cells)
    fig.suptitle(f"{film_title(meta)}: how fast the bulk magnetisation settles, one line per trajectory", fontsize=10)
    _legend_below(fig, axs, ncol=4)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_bulk_vs_cell(d, meta, groups, s1, s_end, path):
    cells = d["cell_nm"]
    rms1, rms, final = bulk_errors(d, meta)
    t_end = meta["ns"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    for name, idx in groups.items():
        c = INK if name == "all" else IC_COLORS[name]
        lw = 2.2 if name == "all" else 1.6
        axs[0, 0].plot(cells, rms1[:, idx].mean(1), color=c, lw=lw, marker="o", ms=4, label=name)
        axs[0, 1].plot(cells, rms[:, idx].mean(1), color=c, lw=lw, marker="o", ms=4, label=name)
        axs[0, 2].plot(cells, final[:, idx].mean(1), color=c, lw=lw, marker="o", ms=4, label=name)
    axs[0, 0].set_ylabel("RMS |<m> - <m>_ref| over 0-1 ns", fontsize=9)
    axs[0, 1].set_ylabel(f"RMS |<m> - <m>_ref| over 0-{t_end:g} ns", fontsize=9)
    axs[0, 2].set_ylabel(f"|<m> - <m>_ref| at {t_end:g} ns", fontsize=9)
    # energies: the coarse solver's and the regridded reference's, over the fine reference's
    steps = d["steps_cmp"]
    for col, term in enumerate(("exchange", "demag", "zeeman")):
        ax = axs[1, col]
        for s, c, when in ((s1, RAMP[1], "1 ns"), (s_end, RAMP[4], f"{t_end:g} ns")):
            fine = d["E_truth_fine"][:, steps[s], col]  # (traj,)
            solver = d["E_pred"][:, :, steps[s], col] / fine[None]
            regrid = d["E_truth_coarse"][:, :, s, col] / fine[None]
            ax.plot(cells, solver.mean(1), color=c, lw=1.8, marker="o", ms=4,
                    label=f"coarse solver at {when}")  # fmt: skip
            ax.plot(cells, regrid.mean(1), color=c, lw=1.2, ls=":", marker="o", ms=3,
                    label=f"reference regridded at {when} (what the mesh can hold)")  # fmt: skip
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_ylabel(f"{term} energy / 5 nm reference's", fontsize=9)
    for ax in axs.ravel():
        _style(ax)
        _cell_axis(ax, cells, top=ax in axs[0])
    axs[1, 0].set_ylim(0, 1.3)
    fig.suptitle(f"{film_title(meta)}: bulk quantities", fontsize=11)
    _legend_below(fig, axs, ncol=4)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_where(d, meta, groups, s1, s_end, path):
    cells = d["cell_nm"]
    M = d["metrics"]
    im = {m: i for i, m in enumerate(METRICS)}
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.2))
    ax = axs[0]
    for s, c, when in ((s1, RAMP[1], "1 ns"), (s_end, RAMP[4], f"{meta['ns']:g} ns")):
        ax.plot(cells, [_nanmean(M[i, :, s, im["mse_textured"]]) for i in range(len(cells))], color=c, lw=1.8, marker="o", ms=4,
                label=f"textured cells (area average lost > {100 * (1 - TEXTURE_NORM):.0f}% norm), {when}")  # fmt: skip
        ax.plot(cells, [_nanmean(M[i, :, s, im["mse_smooth"]]) for i in range(len(cells))], color=c, lw=1.2, ls="--", marker="o", ms=3,
                label=f"smooth cells, {when}")  # fmt: skip
    ax.set_yscale("log")
    ax.set_ylabel("MSE, all trajectories", fontsize=9)
    ax = axs[1]
    for name, idx in groups.items():
        c = INK if name == "all" else IC_COLORS[name]
        ax.plot(cells, M[:, idx, s1, im["textured_frac"]].mean(1), color=c, lw=2.2 if name == "all" else 1.6,
                marker="o", ms=4, label=f"{name}")  # fmt: skip
    ax.set_ylabel("textured cells in the reference at 1 ns (fraction)", fontsize=9)
    ax.set_ylim(0, 1)
    ax = axs[2]
    ax.plot(cells, d["onestep_mse"].mean(axis=(1, 2)), color=RAMP[1], lw=1.8, marker="o", ms=4,
            label="one step from the regridded reference (mean over the first 1 ns)")  # fmt: skip
    ax.plot(cells, M[:, :, s1, im["mse"]].mean(1), color=RAMP[4], lw=1.8, marker="o", ms=4,
            label="free rollout at 1 ns (100 steps)")  # fmt: skip
    ax.plot(cells, d["persist_mse"][:, :, s1].mean(1), color=MUTED, lw=1.2, ls="--", label="persistence at 1 ns")
    ax.set_yscale("log")
    ax.set_ylabel("MSE, all trajectories", fontsize=9)
    for ax in axs:
        _style(ax)
        _cell_axis(ax, cells)
    fig.suptitle(f"{film_title(meta)}: where the error lives, and how it compounds", fontsize=11)
    _legend_below(fig, axs, ncol=3)
    fig.tight_layout(rect=(0, 0.2, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_spectra(d, meta, groups, path):
    cells = d["cell_nm"]
    side_nm = meta["side_um"] * 1e3
    shown = _shown(cells)
    names = list(groups)
    fig, axs = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.8), sharey=True)
    for ax, name in zip(axs, names):
        idx = groups[name]
        for (i, nm), c in zip(shown, RAMP):
            n = int(d["ns"][i])
            pt = d[f"spec_truth_n{n}"][idx, 0].sum(0)  # at 1 ns
            pe = d[f"spec_err_n{n}"][idx, 0].sum(0)
            k = np.arange(1, n // 2 + 1)
            ax.plot(side_nm / k, pe[1:] / pt[1:], color=c, lw=1.8, label=f"{nm:.3g} nm cells")
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_title(f"{name}", fontsize=9)
        ax.set_xlabel("wavelength (nm)", fontsize=8, color=SECOND)
        _style(ax)
    axs[0].set_ylabel("error power / reference power at 1 ns", fontsize=9)
    fig.suptitle(f"{film_title(meta)}: which length scales are wrong (1 = as wrong as the signal)", fontsize=10)
    _legend_below(fig, axs, ncol=len(shown))
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_snapshots(d, meta, s1, path):
    cells = d["cell_nm"]
    ic = meta["ic"]
    moved = d["persist_mse"][0, :, s1]  # how much each trajectory changed over 1 ns
    picks = []
    for want in ("multi-domain", "uniform", "s-state"):
        js = [j for j, x in enumerate(ic) if x == want]
        if js:
            picks.append(max(js, key=lambda j: moved[j]))
    shown = _shown(cells)
    snap_steps = d["snap_steps"]
    times = [(k, snap_steps[k] * meta["dt"] * 1e9) for k in (0, len(snap_steps) - 1)]
    rows = [(j, k, tns) for j in picks for k, tns in times]
    ncol = 1 + len(shown)
    fig, axs = plt.subplots(len(rows), ncol, figsize=(1.9 * ncol, 1.9 * len(rows) + 0.6))
    axs = np.atleast_2d(axs)
    for r, (j, k, tns) in enumerate(rows):
        fields = [("5 nm reference", d["snap_truth_fine"][j, k])]
        fields += [(f"{nm:.3g} nm", d[f"snap_pred_n{int(d['ns'][i])}"][j, k]) for i, nm in shown]
        for ax, (label, m) in zip(axs[r], fields):
            m = m.astype(np.float32)
            ax.imshow(np.degrees(np.arctan2(m[..., 1], m[..., 0])).T, cmap="hsv", vmin=-180, vmax=180,
                      origin="lower", interpolation="nearest")  # fmt: skip
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_color(GRID)
            if r == 0:
                ax.set_title(label, fontsize=8)
        axs[r, 0].set_ylabel(f"traj. {j} ({ic[j]})\n{tns:g} ns", fontsize=7)
    fig.suptitle(f"{film_title(meta)}: in-plane angle of m (hue), reference and coarse micromagnetics", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def film_title(meta):
    return f"{meta['film']} ({meta['side_um']:g} um film)"


# --- summary ----------------------------------------------------------------


def summary(d, meta, groups, s1, s_end):
    cells = d["cell_nm"]
    M = d["metrics"]
    im = {m: i for i, m in enumerate(METRICS)}
    rms1, rms, final = bulk_errors(d, meta)
    t_pred = settle_times(d["bulk_pred"], meta["dt"])
    t_ref = settle_times(d["bulk_truth"], meta["dt"])
    steps = d["steps_cmp"]
    rows = []
    for i, c in enumerate(cells):
        row = {"cell_nm": float(c), "cell_over_l_ex": float(c / L_EX_NM), "cells": int(d["ns"][i]),
               "seconds_per_step": float(np.nanmedian(d["seconds_per_step"][i]))}  # fmt: skip
        for name, idx in groups.items():
            g = {}
            for s, when in ((s1, "1ns"), (s_end, "end")):
                g[f"mse_{when}"] = float(M[i, idx, s, im["mse"]].mean())
                g[f"vrmse_{when}"] = float(M[i, idx, s, im["vrmse"]].mean())
                g[f"ang50_{when}"] = float(M[i, idx, s, im["ang50"]].mean())
                g[f"ang90_{when}"] = float(M[i, idx, s, im["ang90"]].mean())
                g[f"frac_off_{when}"] = float(M[i, idx, s, im["frac_off"]].mean())
                g[f"persist_mse_{when}"] = float(d["persist_mse"][i, idx, s].mean())
                for col, term in enumerate(("exchange", "demag", "zeeman")):
                    fine = d["E_truth_fine"][idx, steps[s], col]
                    g[f"E_{term}_ratio_{when}"] = float((d["E_pred"][i, idx, steps[s], col] / fine).mean())
                    g[f"E_{term}_regrid_ratio_{when}"] = float((d["E_truth_coarse"][i, idx, s, col] / fine).mean())
            g["mse_textured_1ns"] = _nanmean(M[i, idx, s1, im["mse_textured"]])
            g["mse_smooth_1ns"] = _nanmean(M[i, idx, s1, im["mse_smooth"]])
            g["textured_frac_1ns"] = float(M[i, idx, s1, im["textured_frac"]].mean())
            g["onestep_mse"] = float(d["onestep_mse"][i, idx].mean())
            g["bulk_rms_1ns"] = float(rms1[i, idx].mean())
            g["bulk_rms_end"] = float(rms[i, idx].mean())
            g["bulk_final"] = float(final[i, idx].mean())
            g["settle_ns"] = float(t_pred[i, idx].mean())
            g["settle_ratio"] = float((t_pred[i, idx] / t_ref[idx]).mean())
            row[name] = g
        rows.append(row)
    return rows


def print_table(rows, groups):
    print(f"{'cell':>6} {'/l_ex':>5} | {'mse 1ns':>8} {'mse end':>8} {'ang50 1':>7} {'ang50 e':>7} "
          f"{'>30deg 1':>8} {'1step':>8} | {'bulk rms1':>9} {'bulk rmsE':>9} {'bulk fin':>8} {'settle/ref':>10} | {'Eex/ref':>7} {'Edm/ref':>7} | {'ms/st':>5}")  # fmt: skip
    for name in groups:
        print(f"--- {name}")
        for r in rows:
            g = r[name]
            print(f"{r['cell_nm']:6.1f} {r['cell_over_l_ex']:5.1f} | {g['mse_1ns']:8.1e} {g['mse_end']:8.1e} "
                  f"{g['ang50_1ns']:7.1f} {g['ang50_end']:7.1f} {g['frac_off_1ns']:8.2f} {g['onestep_mse']:8.1e} | "
                  f"{g['bulk_rms_1ns']:9.3f} {g['bulk_rms_end']:9.3f} {g['bulk_final']:8.3f} {g['settle_ratio']:10.2f} | "
                  f"{g['E_exchange_ratio_1ns']:7.2f} {g['E_demag_ratio_1ns']:7.2f} | {r['seconds_per_step'] * 1e3:5.1f}")  # fmt: skip


def plot(out_dir, film):
    out_dir = Path(out_dir)
    d, meta = load(out_dir, film)
    groups = _groups(meta["ic"])
    steps = list(d["steps_cmp"])
    n1 = int(round(1e-9 / meta["dt"]))
    s1 = steps.index(n1) if n1 in steps else len(steps) - 1
    s_end = len(steps) - 1
    print(f"{film}: {meta['n_done']} of {len(meta['cell_nm'])} meshes done, {meta['n_traj']} trajectories, {meta['ns']:g} ns")
    rows = summary(d, meta, groups, s1, s_end)
    (out_dir / f"{film}_summary.json").write_text(json.dumps(rows, indent=1))
    print_table(rows, groups)
    fig_pointwise(d, meta, groups, s1, s_end, out_dir / f"{film}_pointwise_vs_cell.png", closure_floor(out_dir))
    fig_error_vs_time(d, meta, groups, out_dir / f"{film}_error_vs_time.png")
    fig_bulk_m(d, meta, out_dir / f"{film}_bulk_m.png")
    fig_bulk_vs_cell(d, meta, groups, s1, s_end, out_dir / f"{film}_bulk_vs_cell.png")
    fig_where(d, meta, groups, s1, s_end, out_dir / f"{film}_where_and_onestep.png")
    fig_spectra(d, meta, groups, out_dir / f"{film}_spectra.png")
    fig_snapshots(d, meta, s1, out_dir / f"{film}_snapshots.png")
    fig_settle(d, meta, out_dir / f"{film}_settle.png")
    t_pred = settle_times(d["bulk_pred"], meta["dt"])
    t_ref = settle_times(d["bulk_truth"], meta["dt"])
    print(f"settling time (ns, within {SETTLE:g} of the final <m>) per trajectory; columns = cell sizes")
    print(f"{'traj':>18} {'ref':>5} " + " ".join(f"{c:5.3g}" for c in d["cell_nm"]))
    for j in range(meta["n_traj"]):
        print(f"{j:>3} {meta['ic'][j]:>14} {t_ref[j]:5.2f} " + " ".join(f"{t:5.2f}" for t in t_pred[:, j]))
    print(f"figures in {out_dir}")
