"""Per-trajectory rollout errors of runs and of the coarse LLG solver on one test film, for
figures that average over chosen trajectories (`datagen/plot_vrmse_simple.py --trajs`).

The scaling cache keeps only trajectory means, so this re-rolls both exactly as
`eval_scaling`'s model and baseline stages do: the runs with their models (JAX) and
magnum.np on the coarse mesh (torch), from the same first frames against the same truth.
Writes `<cache>/pertraj_<film>.npz` with `<arm>_mse_k<k>` and `<arm>_vrmse_k<k>`, each
(n_traj, 100); arm is `solver` or the run with `/` -> `__`. Runs on one MIG slice:

    uv run python -m datagen.pertraj --film sq256 --runs closure/seed_0 closure/seed_1
"""

import argparse
from pathlib import Path

import numpy as np

from datagen.eval_scaling import DT, FILMS, MATERIAL, N_STEPS, cache_dir, load_truth
from paths import RESULTS


def per_trajectory(pred, truth, eps=1e-7):
    """(n_traj, T) MSE and the well's VRMSE of (n_traj, T, nx, ny, 3) against truth."""
    err = ((pred - truth) ** 2).mean(axis=(2, 3))
    var = truth.std(axis=(2, 3), ddof=1) ** 2
    return err.mean(axis=2), np.sqrt(err / (var + eps)).mean(axis=2)


def solver_rollouts(truth, hs, dx):
    import torch
    from magnumnp import DemagField, ExchangeField, ExternalField, LLGSolver, Mesh, State, set_log_level

    set_log_level(250)
    preds = []
    for m0, h in zip(truth[:, 0], hs):
        nx, ny, _ = m0.shape
        state = State(Mesh((nx, ny, 1), dx))
        state.material = dict(MATERIAL)
        state.m = torch.as_tensor(m0[:, :, None, :], dtype=torch.get_default_dtype()).to(
            torch.get_default_device()
        )
        llg = LLGSolver([DemagField(), ExchangeField(), ExternalField([float(x) for x in h])])
        frames = []
        for _ in range(N_STEPS):
            llg.step(state, DT)
            frames.append(state.m.squeeze(2).cpu().numpy().astype(np.float32))
        preds.append(np.stack(frames))
    return np.stack(preds)


def model_rollouts(run, dataset, truth, hs, dx):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from model import load_config, load_model
    from physics import mesh_physics
    from utils import conditioning, rollout

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    path = RESULTS / dataset / run
    model = eqx.nn.inference_mode(load_model(path, jr.PRNGKey(0), "model"))
    config = load_config(path)
    n_in = config.in_frames
    n_traj, _, nx, ny, _ = truth.shape
    x = (jnp.arange(nx) + 0.5) * dx[0]
    y = (jnp.arange(ny) + 0.5) * dx[1]
    grid1 = jnp.stack(jnp.meshgrid(x, y, indexing="ij"), axis=-1)[None]
    physics = mesh_physics(config, (nx, ny), dx)
    scalars = np.zeros((n_traj, 6), np.float32)
    scalars[:, 0], scalars[:, 1], scalars[:, 2] = MATERIAL["Ms"], MATERIAL["A"], MATERIAL["alpha"]
    scalars[:, 3:6] = hs
    bs = int(max(1, min(n_traj, 8 * 256**2 // (nx * ny))))
    preds = []
    for i in range(0, n_traj, bs):
        sl = slice(i, min(i + bs, n_traj))
        b = sl.stop - sl.start
        grid = jnp.repeat(grid1, b, axis=0)
        cond = conditioning(jnp.asarray(scalars[sl]), grid)
        pred = rollout(model, jnp.asarray(truth[sl, :n_in]), cond, grid, N_STEPS - (n_in - 1),
                       jnp.asarray(hs[sl], jnp.float32), physics)  # fmt: skip
        preds.append(np.concatenate([truth[sl, 1:n_in], np.asarray(pred)], axis=1))
    return np.concatenate(preds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--film", default="sq256", choices=list(FILMS))
    p.add_argument("--runs", nargs="+", default=("closure/seed_0", "closure/seed_1"))
    args = p.parse_args()
    cache = cache_dir(args.dataset)
    out_path = cache / f"pertraj_{args.film}.npz"
    out = dict(np.load(out_path)) if out_path.exists() else {}
    for k in FILMS[args.film][1]:
        truth, hs, dx = load_truth(cache, args.film, k)
        ref = truth[:, 1:]
        arms = {}
        if k > 1 and f"solver_mse_k{k}" not in out:
            arms["solver"] = solver_rollouts(truth, hs, dx)
        for run in args.runs:
            arm = run.replace("/", "__")
            if f"{arm}_mse_k{k}" not in out:
                arms[arm] = model_rollouts(run, args.dataset, truth, hs, dx)
        for arm, pred in arms.items():
            mse, vrmse = per_trajectory(pred, ref)
            out[f"{arm}_mse_k{k}"], out[f"{arm}_vrmse_k{k}"] = mse, vrmse
            print(f"{args.film} k={k} {arm}: per-trajectory mean MSE", np.round(mse.mean(1), 4), flush=True)
        np.savez(out_path, **out)
    print(out_path)


if __name__ == "__main__":
    main()
