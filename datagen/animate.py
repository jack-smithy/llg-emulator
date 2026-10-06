"""Animations like a run's `ref.gif` / `pred.gif` / `err.gif`, on larger films and coarser
cells, for a constant-field test trajectory and for a hysteresis sweep.

    uv run python -m datagen.animate rollout --run k1-long/seed_0 --film sq512 --traj 4 --cells 5 10 20 40
    uv run python -m datagen.animate rollout --run k1-long/seed_0 --film large --traj 0 --cells 20 40
    uv run python -m datagen.animate loop --run k1-long/seed_0 --film sq512 --cells 5 10 20 40 --axis x

`rollout` (GPU): the film's trajectory area-averaged onto each cell size is the reference;
the run and the coarse LLG solver (`physics.LLGStepper`, the reference itself at 5 nm) are
rolled out from its first frame under its field for the data's 100 steps. Written to
`RESULTS/<dataset>/<run>/animations/rollout_<film>_traj<j>_<cell>nm_{ref,pred,solver,err,cos,cos_solver}.gif`
(err = pred - ref per component; cos = the cosine similarity to the reference per cell, one
panel, for the run and for the solver). Frames are strided down to at most `MAX_PX` pixels a side.

`loop` (CPU): the snapshots `datagen/hysteresis.py run --film <film> --snap-mt ...` stored for
the reference (5 nm), the coarse solver and the run, as
`loop_<film>_<axis>_<cell>nm_{ref,solver,pred,cos,cos_solver}.gif`, the applied field in the
title; the cosine similarity is against the 5 nm reference area-averaged onto the mesh.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from datagen.eval_scaling import FILMS, N_STEPS, block_mean, cache_dir, load_truth, trajectories
from datagen.generate_varied_field import DT, DX, MATERIAL
from paths import RESULTS

MU_0 = 4e-7 * np.pi
MAX_PX = 384
LABELS = ("$m_x$", "$m_y$", "$m_z$")


def animate(frames, titles, path, cmap="RdBu_r", vlim=1.0, interval=100):
    """frames: (T, H, W, 3) -> a GIF of the three components side by side, one title per
    frame; strided to MAX_PX."""
    s = max(1, int(np.ceil(frames.shape[1] / MAX_PX)))
    frames = np.asarray(frames[:, ::s, ::s], dtype=np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.9))
    ims = []
    for c, ax in enumerate(axes):
        ims.append(ax.imshow(frames[0, ..., c].T, cmap=cmap, vmin=-vlim, vmax=vlim, origin="lower"))
        ax.set_title(LABELS[c], fontsize=9)
        ax.axis("off")
    text = fig.suptitle(titles[0], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(t):
        for c, im in enumerate(ims):
            im.set_data(frames[t, ..., c].T)
        text.set_text(titles[t])
        return ims + [text]

    FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False).save(path, writer="pillow")
    plt.close(fig)
    print(path, flush=True)


def animate_cos(pred, ref, titles, path, interval=100):
    """One panel: the cosine similarity pred . ref per cell (both unit vectors, so the
    magnitude carries no error), 1 = identical, 0 = 90 degrees off, clipped below 0."""
    cos = np.einsum("thwc,thwc->thw", pred.astype(np.float32), ref.astype(np.float32))
    s = max(1, int(np.ceil(cos.shape[1] / MAX_PX)))
    cos = cos[:, ::s, ::s]
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    im = ax.imshow(cos[0].T, cmap="Blues_r", vmin=0.0, vmax=1.0, origin="lower")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="cosine similarity to the reference")
    text = fig.suptitle(titles[0], fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    def update(t):
        im.set_data(cos[t].T)
        text.set_text(titles[t])
        return [im, text]

    FuncAnimation(fig, update, frames=len(cos), interval=interval, blit=False).save(path, writer="pillow")
    plt.close(fig)
    print(path, flush=True)


def reference_frames(cache, film, k, traj):
    """(101, nx, ny, 3) float32 of one trajectory at factor k and its field: the cache's
    truth where it has this factor, else the dataset's frames area-averaged on the fly."""
    if (cache / f"truth_{film}_k{k}.npy").exists():
        truth, hs, dx = load_truth(cache, film, k)
        return truth[traj], np.asarray(hs[traj]), dx
    for j, (read, h) in enumerate(trajectories(FILMS[film][0])):
        if j == traj:
            frames = np.stack([block_mean(read(t), k) for t in range(N_STEPS + 1)]).astype(np.float32)
            return frames, np.asarray(h), (DX[0] * k, DX[1] * k, DX[2])
    raise SystemExit(f"{film} has no trajectory {traj}")


def stage_rollout(out_dir, run, dataset, film, traj, cells):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from model import load_config, load_model
    from physics import mesh_physics, stepper_cache
    from utils import conditioning, rollout

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    cache = cache_dir(dataset)
    path = RESULTS / dataset / run
    config = load_config(path)
    model = eqx.nn.inference_mode(load_model(path, jr.PRNGKey(0), "model"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for nm in cells:
        k = int(round(nm / (DX[0] * 1e9)))
        ref, h, dx = reference_frames(cache, film, k, traj)
        nx, ny = ref.shape[1:3]
        x = (jnp.arange(nx) + 0.5) * dx[0]
        y = (jnp.arange(ny) + 0.5) * dx[1]
        grid = jnp.stack(jnp.meshgrid(x, y, indexing="ij"), axis=-1)[None]
        scalars = jnp.asarray([[MATERIAL["Ms"], MATERIAL["A"], MATERIAL["alpha"], *h]], jnp.float32)
        physics = mesh_physics(config, (nx, ny), dx)
        pred = np.asarray(rollout(model, jnp.asarray(ref[None, :1]), conditioning(scalars, grid), grid,
                                  N_STEPS, jnp.asarray(h[None], jnp.float32), physics)[0])  # fmt: skip
        titles = [f"{film}, trajectory {traj}, {nm:g} nm cells, t = {t * DT * 1e9:.2f} ns" for t in range(1, N_STEPS + 1)]
        tag = f"rollout_{film}_traj{traj}_{nm:g}nm"
        animate(ref[1:], [t + ": reference" for t in titles], out_dir / f"{tag}_ref.gif")
        animate(pred, [t + f": {run}" for t in titles], out_dir / f"{tag}_pred.gif")
        animate(pred - ref[1:], [t + ": prediction - reference" for t in titles], out_dir / f"{tag}_err.gif", vlim=0.5)
        animate_cos(pred, ref[1:], [t + f": {run}" for t in titles], out_dir / f"{tag}_cos.gif")
        if k > 1:
            solver = stepper_cache((nx, ny), dx)
            step = eqx.filter_jit(lambda m, h, s=solver: s(m, h))
            m = jnp.asarray(np.moveaxis(ref[0], -1, 0))
            hj = jnp.asarray(h, jnp.float32)
            frames = []
            for _ in range(N_STEPS):
                m = step(m, hj)
                frames.append(np.moveaxis(np.asarray(m), 0, -1))
            frames = np.stack(frames)
            animate(frames, [t + ": LLG solver" for t in titles], out_dir / f"{tag}_solver.gif")
            animate_cos(frames, ref[1:], [t + ": LLG solver" for t in titles], out_dir / f"{tag}_cos_solver.gif")
        del pred, ref


def stage_loop(out_dir, run, dataset, film, cells, axis):
    from datagen.hysteresis import AXES

    hdir = RESULTS / dataset / ("hysteresis" if film == "sq256" else f"hysteresis_{film}")
    meta = json.loads((hdir / "config.json").read_text())
    solver = dict(np.load(hdir / "solver.npz"))
    arm = dict(np.load(hdir / f"{run.replace('/', '__')}.npz"))
    c = AXES[axis]
    h_mt = solver[f"H_{axis}"][:, c] * MU_0 * 1e3
    out_dir.mkdir(parents=True, exist_ok=True)
    for nm in cells:
        snaps = {"ref": solver[f"snap_{axis}_5"], "pred": arm[f"snap_{axis}_{nm:g}"]}
        if nm != 5:
            snaps["solver"] = solver[f"snap_{axis}_{nm:g}"]
        n_snap = snaps["pred"].shape[0]
        # snapshot i was taken after i * (steps per snapshot) steps
        idx = np.minimum(np.arange(n_snap) * (len(h_mt) // (n_snap - 1)), len(h_mt) - 1)
        names = {"ref": "reference (5 nm)", "pred": run, "solver": "LLG solver"}
        k = int(round(nm / (DX[0] * 1e9)))
        ref_coarse = np.stack([block_mean(f, k) for f in np.moveaxis(snaps["ref"].astype(np.float32), 1, -1)])
        for key, arr in snaps.items():
            frames = np.moveaxis(arr.astype(np.float32), 1, -1)  # (T, n, n, 3)
            titles = [f"{film}, {nm:g} nm cells, mu0 H_{axis} = {h_mt[i]:+.1f} mT: {names[key]}" for i in idx]
            animate(frames, titles, out_dir / f"loop_{film}_{axis}_{nm:g}nm_{key}.gif", interval=150)
            if key != "ref":  # against the reference area-averaged onto this mesh
                animate_cos(frames, ref_coarse, titles, out_dir / f"loop_{film}_{axis}_{nm:g}nm_cos{'' if key == 'pred' else '_solver'}.gif", interval=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("rollout", "loop"))
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--run", required=True)
    p.add_argument("--film", default="sq512")
    p.add_argument("--traj", type=int, default=4)
    p.add_argument("--cells", type=float, nargs="+", default=(5, 10, 20, 40))
    p.add_argument("--axis", default="x", choices=("x", "y"))
    args = p.parse_args()
    out_dir = RESULTS / args.dataset / args.run / "animations"
    if args.stage == "rollout":
        stage_rollout(out_dir, args.run, args.dataset, args.film, args.traj, args.cells)
    else:
        stage_loop(out_dir, args.run, args.dataset, args.film, args.cells, args.axis)


if __name__ == "__main__":
    main()
