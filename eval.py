import json
from argparse import ArgumentParser
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import torch
from the_well.benchmark.metrics import MSE, VRMSE
from the_well.data import WellDataset

from datagen.generate_varied_field import DX
from model import load_config, load_model
from paths import RESULTS
from physics import mesh_physics
from plot import animate_channels, plot_learning_curves, plot_norms, plot_rollout
from utils import (
    conditioning,
    numpy_collate,
    one_step_preds,
    relative_norm_error,
    rollout,
)

jax.config.update("jax_compilation_cache_dir", ".jax_cache")

OUT_FRAMES = 1
NUM_WORKERS = 1
N_FRAMES_ROLLOUT = 100
ROLLOUT_IDX = 7


def main(seed, configuration, dataset):
    path = f"datasets/{dataset}"
    results_path = RESULTS / dataset / configuration / f"seed_{seed}"

    with open(results_path / "stats.json", "r") as f:
        stats = json.load(f)

    # rebuilt from model.eqx + metadata.json alone; the key only seeds the
    # skeleton whose weights are then overwritten
    model = load_model(results_path, key=jr.PRNGKey(seed), tag="model")
    model = eqx.nn.inference_mode(model)
    config = load_config(results_path)
    IN_FRAMES = config.in_frames  # context frames the model takes
    print("model loaded")

    ### validation
    val_dataset = WellDataset(
        path=path,
        well_split_name="valid",
        n_steps_input=IN_FRAMES,
        n_steps_output=OUT_FRAMES,
        use_normalization=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=stats["config"]["batch_size"],
        num_workers=NUM_WORKERS,
        collate_fn=numpy_collate,
    )
    # the coarse solver / demag a model uses, on the native mesh
    physics = mesh_physics(config, val_dataset.metadata.spatial_resolution, DX)
    pred, truth = one_step_preds(model, loader, physics)
    stats["metrics"] = {
        "mse": MSE()(pred, truth, val_dataset.metadata).mean().item(),
        "vrmse": VRMSE()(pred, truth, val_dataset.metadata).mean().item(),
    }
    with open(results_path / "stats.json", "w") as f:
        json.dump(stats, f)
    print(f"one-step metrics: {stats['metrics']}")

    ### free rollout of one validation trajectory
    rollout_dataset = WellDataset(
        path=path,
        well_split_name="valid",
        n_steps_input=N_FRAMES_ROLLOUT,
        n_steps_output=OUT_FRAMES,
        use_normalization=False,
    )

    sample = numpy_collate([rollout_dataset[ROLLOUT_IDX]])
    truth = sample["input_fields"]  # (1, N_FRAMES_ROLLOUT, Lx, Ly, F)
    cond = conditioning(sample["constant_scalars"], sample["space_grid"])
    pred = rollout(
        model,
        truth[:, :IN_FRAMES],
        cond,
        sample["space_grid"],
        n_steps=truth.shape[1] - IN_FRAMES,
        h=sample["constant_scalars"][:, 3:6],
        physics=physics,
    )
    truth = truth[:, IN_FRAMES:]

    ref, pred = truth[0], np.asarray(pred[0])
    np.save(results_path / "rollout.npy", np.stack([ref, pred]))
    print("saved rollout")

    ### plots
    fig, _ = plot_learning_curves(stats=stats)
    fig.savefig(results_path / "learning_curve.png")
    plt.close()
    print("plotted learning curve")

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
    print("plotted rollout animations")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    main(
        seed=args.seed,
        configuration=args.configuration,
        dataset=args.dataset,
    )
