import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

from plot import animate_channels, plot_learning_curves, plot_norms, plot_rollout
from utils import relative_norm_error


def main(seed, configuration):
    base_path = Path("results/permalloy_varied_field")
    results_path = base_path / configuration / f"seed_{seed}"

    with open(results_path / "stats.json", "r") as f:
        stats = json.load(f)

    fig, _ = plot_learning_curves(stats=stats)
    fig.savefig(results_path / "learning_curve.png")
    plt.close()
    print("plotted learning curve")

    ref, pred = np.load(results_path / "rollout.npy")
    ref_bulk, pred_bulk = np.mean(ref, axis=(1, 2)), np.mean(pred, axis=(1, 2))

    fig, _ = plot_rollout(ref_bulk=ref_bulk, pred_bulk=pred_bulk)
    fig.savefig(results_path / "rollout.png")
    plt.close()
    print("plotted rollout")

    ref_norm, pred_norm = relative_norm_error(ref), relative_norm_error(pred)
    fig, _ = plot_norms(ref_norm=ref_norm, pred_norm=pred_norm)
    fig.savefig(results_path / "norms.png")
    print("plotted norms")

    anim_ref = animate_channels(ref)
    anim_ref.save(results_path / "ref.gif", writer="pillow")
    plt.close()

    anim_pred = animate_channels(pred)
    anim_pred.save(results_path / "pred.gif", writer="pillow")
    plt.close()

    anim_err = animate_channels(ref - pred)
    anim_err.save(results_path / "err.gif", writer="pillow")
    plt.close()
    print("plotted sp4 animations")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", type=str, required=True)
    args = parser.parse_args()
    main(seed=args.seed, configuration=args.configuration)
