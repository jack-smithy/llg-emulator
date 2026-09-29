"""Roll a trained emulator out on the held-out film geometries.

Runs in the project env, from the repo root:
`uv run python -m datagen.eval_geometries --seed 2 --arch fno --configuration test --dataset llg_field_switching`;
`scripts/eval_geometries.slrm` runs it on one MIG slice.

The model is fully convolutional and takes the applied field as constant channels and
the film's cell-centre (x, y) coordinates in um as two input channels (`utils.with_coords`,
from the sample's own `space_grid`, so every geometry sits on the training set's real-space
scale), so weights trained on 256x256 films run on any grid unchanged. Every well root under
`datasets/<dataset>/geometries/<name>` (written by `datagen/generate_geometries.py`) is
evaluated, plus the 256x256 `test` split of the training dataset itself, which shares
the same 8 seeds and so the same applied fields: the in-distribution reference. Per
geometry, as `eval.py` does on the validation split:

- teacher-forced one-step MSE and VRMSE over every (t, t+1) pair, accumulated batch by
  batch so a 1024x1024 film never sits in memory whole;
- a free rollout over the whole trajectory from frame 0 for every sample, reduced to
  the per-step MSE and the bulk magnetisation <m>(t) of prediction and reference.

Written to `results/<dataset>/<arch>/<configuration>/seed_<seed>/geometries/`:

    metrics.json          per geometry: cells, one-step mse/vrmse, rollout mse per step
                          (mean over samples and at the last step), bulk <m>(t) per
                          sample, and wall-clock seconds per rollout step
    rollout_mse.png       rollout MSE against step, one line per geometry
    rollout_<name>.png    bulk <m>(t), reference against prediction, for one sample

`metrics.json` is rewritten after every geometry, so a run cut short still leaves the
geometries it finished. A geometry the architecture cannot take (an axis shorter than
the FNO's modes, say) is skipped with a message rather than ending the run.
"""

import json
import time
from argparse import ArgumentParser
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from the_well.benchmark.metrics import MSE, VRMSE
from the_well.data import WellDataset
from tqdm import tqdm

from model import load_model
from plot import plot_rollout, plot_rollout_mse
from utils import conditioning, numpy_collate, predict, prepare_batch, rollout

jax.config.update("jax_compilation_cache_dir", ".jax_cache")

IN_FRAMES = 1  # the model is a one-step map m_t -> m_{t+1}
OUT_FRAMES = 1
N_FRAMES_ROLLOUT = 100  # the whole trajectory: one context frame, 99 predicted
NUM_WORKERS = 1
ROLLOUT_IDX = 7  # the sample whose bulk <m>(t) is plotted, as in eval.py
REFERENCE = "sq256"  # the training film, i.e. the dataset's own test split


def one_step_metrics(model, root, batch_size):
    """Teacher-forced one-step MSE / VRMSE over every pair, one batch at a time.

    returns: (mse, vrmse, cells) — scalars averaged over pairs and fields, and the
    film's (nx, ny).
    """
    dataset = WellDataset(
        path=root,
        well_split_name="test",
        n_steps_input=IN_FRAMES,
        n_steps_output=OUT_FRAMES,
        use_normalization=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        collate_fn=numpy_collate,
    )
    mse, vrmse = [], []
    for batch in tqdm(loader, mininterval=10):
        m0, m1, cond = prepare_batch(batch)
        pred, truth = (
            rearrange(x, "B F Lx Ly -> B 1 Lx Ly F")
            for x in (np.asarray(predict(model, m0, cond)), m1)
        )
        # the well's metrics reduce over space and keep (B, 1, F); keep one per sample
        mse.append(MSE()(pred, truth, dataset.metadata).mean(dim=(1, 2)))
        vrmse.append(VRMSE()(pred, truth, dataset.metadata).mean(dim=(1, 2)))
    return (
        torch.cat(mse).mean().item(),
        torch.cat(vrmse).mean().item(),
        tuple(dataset.metadata.spatial_resolution),
    )


def rollouts(model, root):
    """Free rollout of every trajectory from its first frame.

    returns: (mse, bulk_ref, bulk_pred, seconds) — per-step MSE `(n_traj, T)`, bulk
    <m>(t) `(n_traj, T, 3)` for reference and prediction, and each rollout's wall-clock
    seconds (the first includes jit compilation for this grid).
    """
    dataset = WellDataset(
        path=root,
        well_split_name="test",
        n_steps_input=N_FRAMES_ROLLOUT,
        n_steps_output=OUT_FRAMES,
        use_normalization=False,
    )
    mse, bulk_ref, bulk_pred, seconds = [], [], [], []
    for j in tqdm(range(len(dataset))):
        sample = numpy_collate([dataset[j]])
        truth = sample["input_fields"]  # (1, N_FRAMES_ROLLOUT, Lx, Ly, F)
        cond = conditioning(sample["constant_scalars"])
        t0 = time.perf_counter()
        pred = np.asarray(  # blocks until the rollout is done
            rollout(
                model,
                truth[:, :IN_FRAMES],
                cond,
                sample["space_grid"],
                n_steps=truth.shape[1] - IN_FRAMES,
            )
        )
        seconds.append(time.perf_counter() - t0)
        ref = truth[:, IN_FRAMES:]
        mse.append(MSE()(pred, ref, dataset.metadata)[0].mean(dim=-1).numpy())  # (T,)
        bulk_ref.append(ref[0].mean(axis=(1, 2)))
        bulk_pred.append(pred[0].mean(axis=(1, 2)))
    return np.stack(mse), np.stack(bulk_ref), np.stack(bulk_pred), seconds


def main(seed, configuration, dataset, arch, batch_size, geometries):
    data_root = Path("datasets") / dataset
    results_path = Path("results") / dataset / arch / configuration / f"seed_{seed}"
    out = results_path / "geometries"
    out.mkdir(exist_ok=True)

    # rebuilt from model.eqx + metadata.json alone; the key only seeds the skeleton
    model = load_model(results_path, key=jr.PRNGKey(seed), tag="model")
    model = eqx.nn.inference_mode(model)

    roots = {REFERENCE: data_root}
    roots |= {
        p.name: p
        for p in sorted((data_root / "geometries").iterdir())
        if (p / "data" / "test" / "llg_test.hdf5").exists()
    }
    if geometries:
        roots = {name: roots[name] for name in geometries}

    metrics = {}
    for name, root in roots.items():
        print(f"=== {name}", flush=True)
        try:
            mse, vrmse, cells = one_step_metrics(model, str(root), batch_size)
            roll_mse, bulk_ref, bulk_pred, seconds = rollouts(model, str(root))
        except ValueError as e:
            # the grid does not fit the architecture: an axis shorter than the FNO's
            # modes, or not divisible by a hierarchical model's levels
            print(f"skipping {name}: {e}", flush=True)
            continue
        n_steps = roll_mse.shape[1]
        metrics[name] = {
            "cells": list(cells),
            "one_step": {"mse": mse, "vrmse": vrmse},
            "rollout": {
                "mse": roll_mse.mean(axis=0).tolist(),
                "mse_final": float(roll_mse[:, -1].mean()),
                "bulk_ref": bulk_ref.tolist(),
                "bulk_pred": bulk_pred.tolist(),
                # after the first rollout, which also compiles for this grid
                "seconds_per_step": float(np.mean(seconds[1:] or seconds) / n_steps),
            },
        }
        print(
            f"{name} {cells[0]}x{cells[1]}: one-step mse {mse:.3e} vrmse {vrmse:.3f}, "
            f"rollout mse mean {roll_mse.mean():.3e} final {roll_mse[:, -1].mean():.3e}, "
            f"{metrics[name]['rollout']['seconds_per_step'] * 1e3:.1f} ms/step",
            flush=True,
        )
        j = ROLLOUT_IDX % len(bulk_ref)
        fig, _ = plot_rollout(ref_bulk=bulk_ref[j], pred_bulk=bulk_pred[j])
        fig.savefig(out / f"rollout_{name}.png")
        plt.close(fig)
        (out / "metrics.json").write_text(json.dumps(metrics))

    by_cells = sorted(metrics, key=lambda k: np.prod(metrics[k]["cells"]))
    fig, _ = plot_rollout_mse(
        {
            f"{k} {'x'.join(map(str, metrics[k]['cells']))}": np.asarray(
                metrics[k]["rollout"]["mse"]
            )
            for k in by_cells
        }
    )
    fig.savefig(out / "rollout_mse.png")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--configuration", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--geometries", nargs="*", help="a subset of names to run; default is all"
    )
    args = parser.parse_args()
    main(
        seed=args.seed,
        configuration=args.configuration,
        dataset=args.dataset,
        arch=args.arch,
        batch_size=args.batch_size,
        geometries=args.geometries,
    )
