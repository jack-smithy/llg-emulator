"""`rollout.png` with the coarse solver: bulk <m_x>, <m_y>, <m_z> over one 100-step rollout
of one trajectory, one row per cell size, reference against the run and the coarse LLG
solver, from the scaling cache's rollouts (`model_<film>_k<k>.npz`, `baseline_...`).

    uv run python -m datagen.plot_rollout_cells --run k1-wide/seed_0 [--film valid256 --traj 7]

Trajectory 7 of the validation films is the one `eval.py` draws in `rollout.png`. Written
to the run's directory as `rollout_cells_<film>_traj<j>.png`.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datagen.eval_scaling import FILMS, cache_dir, run_dir
from datagen.scaling_report import CELL_NM
from paths import RESULTS

INK, CLOSURE, SOLVER, GRID, SECOND = "#0b0b0b", "#2a78d6", "#eb6834", "#e4e3df", "#52514e"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--run", required=True)
    p.add_argument("--film", default="valid256", choices=list(FILMS))
    p.add_argument("--traj", type=int, default=7)
    args = p.parse_args()
    cache = cache_dir(args.dataset)
    ks = FILMS[args.film][1]
    fig, axs = plt.subplots(len(ks), 3, figsize=(9, 2.1 * len(ks)), sharex=True, sharey=True)
    for r, k in enumerate(ks):
        d = np.load(run_dir(cache, args.run) / f"model_{args.film}_k{k}.npz")
        ref, pred = d["bulk_ref"][args.traj], d["bulk_pred"][args.traj]
        base = cache / f"baseline_{args.film}_k{k}.npz"
        solver = np.load(base)["bulk_pred"][args.traj] if base.exists() else None
        t = np.arange(1, ref.shape[0] + 1)
        for c, ax in enumerate(axs[r]):
            ax.plot(t, ref[:, c], color=INK, lw=2.0, label="reference (5 nm, averaged)")
            if solver is not None:
                ax.plot(t, solver[:, c], color=SOLVER, lw=1.6, label="LLG Solver")
            ax.plot(t, pred[:, c], color=CLOSURE, lw=1.6, label=args.run)
            ax.set_ylim(-1, 1)
            ax.grid(True, color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=SECOND, labelsize=8)
            if r == 0:
                ax.set_title(f"<m_{'xyz'[c]}>", fontsize=10)
            if r == len(ks) - 1:
                ax.set_xlabel("rollout step (10 ps)", fontsize=8, color=SECOND)
        axs[r, 0].set_ylabel(f"{CELL_NM * k:g} nm cells" + (" (native)" if k == 1 else ""), fontsize=9)
    handles, labels = axs[-1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{args.film} trajectory {args.traj}: bulk magnetisation over one rollout", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = RESULTS / args.dataset / args.run / f"rollout_cells_{args.film}_traj{args.traj}.png"
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
