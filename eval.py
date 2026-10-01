import json
from argparse import ArgumentParser
from itertools import islice
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import torch
from the_well.benchmark.metrics import MSE, VRMSE
from the_well.data import WellDataset

from einops import rearrange

from datagen.generate_varied_field import DX
from model import load_model
from physics import demag_cache, llg_cache
from plot import animate_channels, plot_learning_curves, plot_norms, plot_rollout
from utils import (
    conditioning,
    downsample,
    numpy_collate,
    one_step_preds,
    relative_norm_error,
    resampled_mesh,
    rollout,
)

jax.config.update("jax_compilation_cache_dir", ".jax_cache")

IN_FRAMES = 1  # the model is a one-step map m_t -> m_{t+1}
OUT_FRAMES = 1
NUM_WORKERS = 1
N_FRAMES_ROLLOUT = 100
ROLLOUT_IDX = 7


def main(
    seed,
    configuration,
    dataset,
    arch,
    results_root=Path("results"),
    group="",
    max_batches=None,
    pool_factor=1,
):
    path = f"datasets/{dataset}"
    results_path = (
        results_root / dataset / group / arch / configuration / f"seed_{seed}"
    )

    with open(results_path / "stats.json", "r") as f:
        stats = json.load(f)

    # rebuilt from model.eqx + metadata.json alone; the key only seeds the
    # skeleton whose weights are then overwritten
    model, model_config = load_model(results_path, key=jr.PRNGKey(seed), tag="model")
    model = eqx.nn.inference_mode(model)
    print("model loaded")

    ### validation
    val_dataset = WellDataset(
        path=path,
        well_split_name="valid",
        n_steps_input=IN_FRAMES,
        n_steps_output=OUT_FRAMES,
        use_normalization=False,
    )
    # the mesh the model runs on: the dataset's, resampled by --pool-factor
    k = pool_factor
    demag = (
        demag_cache(*resampled_mesh(val_dataset.metadata.spatial_resolution, DX, k))
        if model_config.use_demag
        else None
    )
    solver = (
        llg_cache(*resampled_mesh(val_dataset.metadata.spatial_resolution, DX, k))
        if model_config.use_solver
        else None
    )
    loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=stats["config"]["batch_size"],
        num_workers=NUM_WORKERS,
        collate_fn=numpy_collate,
    )
    # --max-batches (smoke tests) scores a subset of the split instead of all of it
    pred, truth = one_step_preds(
        model,
        islice(loader, max_batches) if max_batches else loader,
        demag,
        k,
        model_config.coords,
        model_config.cell_size_cond,
        solver,
    )
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
    # the reference lives on the pooled mesh too: rollout and truth compare there
    frames = rearrange(truth, "B T Lx Ly F -> (B T) F Lx Ly")
    frames, grid = downsample(frames, sample["space_grid"], k)
    truth = np.asarray(rearrange(frames, "(B T) F Lx Ly -> B T Lx Ly F", B=1))
    cond = conditioning(
        sample["constant_scalars"], grid if model_config.cell_size_cond else None
    )
    pred = rollout(
        model,
        truth[:, :IN_FRAMES],
        cond,
        grid,
        n_steps=truth.shape[1] - IN_FRAMES,
        demag=demag,
        coords=model_config.coords,
        solver=solver,
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
    parser.add_argument("--arch", type=str, required=True)
    # evaluate on --pool-factor-times larger cells, any real >= 1 (match a
    # factor the model trained on)
    parser.add_argument("--pool-factor", type=float, default=1.0)
    # for smoke tests (test.py): read/write a disposable root, score fewer batches
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    main(
        seed=args.seed,
        configuration=args.configuration,
        dataset=args.dataset,
        arch=args.arch,
        results_root=args.results_root,
        group=args.group,
        max_batches=args.max_batches,
        pool_factor=args.pool_factor,
    )
