"""Plot the valid and test loops of runs in results/switch (from their loops.json) against the data, and
list the switching fields per loop branch.

    python plot_switch_runs.py RUN [RUN ...]      (e.g. base_s0 soft1_s0)

writes results/switch/loops_<runs>.png.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import rollout
import train
import train_switch

COLOURS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]


def switching_fields(b_axis: np.ndarray, loop: np.ndarray) -> list[float]:
    """Fields [mT] of the steps where <m> along the axis jumps by more than 0.1."""
    jumps = np.abs(np.diff(loop)) > 0.1
    return [round(float(b), 1) for b in b_axis[1:][jumps]]


def main():
    runs = sys.argv[1:]
    train.COARSEN = train_switch.COARSEN
    reports = {run: json.loads(Path("results/switch", run, "loops.json").read_text()) for run in runs}
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.5), sharey=True, layout="constrained")
    for row, split in enumerate(("valid", "test")):
        m_data, fields, axes = rollout.load_trajectories(split)
        for i, ax in enumerate(axs[row]):
            b_axis = fields[i].cpu().numpy() @ axes[i] * train.mu_0 * 1e3  # [mT]
            data = rollout.mean_along(m_data[i], axes[i])
            ax.plot(b_axis, data, color="#222222", linewidth=1.6, label="data")
            print(f"{split} {i} data: {switching_fields(b_axis, data)}")
            for run, colour in zip(runs, COLOURS):
                loop = np.array(reports[run][split]["loops"][i])
                error = np.abs(loop - data).mean()
                ax.plot(b_axis, loop, color=colour, linewidth=1.2, linestyle="--", label=f"{run}")
                print(f"{split} {i} {run} ({error:.3f}): {switching_fields(b_axis, loop)}")
            ax.set_title(f"{split} {i}: axis {np.degrees(np.arctan2(axes[i][1], axes[i][0])):.0f}°")
            ax.set_xlabel("μ0 H along axis (mT)")
    axs[0, 0].set_ylabel("<m> along axis")
    axs[1, 0].set_ylabel("<m> along axis")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False)
    path = Path("results/switch", f"loops_{'_'.join(runs)}.png")
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
