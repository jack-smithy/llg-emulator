"""Standalone evaluation of a saved checkpoint.

Two kinds of autoregressive rollout:
  * the **benchmark variants** (`data/val/<variant>`, one trajectory each, every
    one on a mesh the model never trained on) — bulk magnetization ⟨m⟩(t) vs. the
    reference plus per-frame correlation. `sp4` is 100x25 cells under an
    out-of-distribution field; `large` is 2000x2000, 61x the training area, and
    is the domain-size check. Together they are the mesh-invariance evidence.
  * per-frame correlation over held-out val trajectories on the training mesh.

The checkpoint is self-describing (`metadata.json` carries the mesh), so nothing
here needs to know how the model was trained.
"""

from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.vtk import VtkGroup
from tqdm import tqdm

from llg_emulator.config import (
    BENCHMARK_VARIANTS,
    JAX_CACHE_DIR,
    benchmark_path,
    sp4_path,
    variant_dirs,
)
from llg_emulator.data import _frames_to_cf, open_trajectory, sample_dirs
from llg_emulator.io import load_model, write_vtr
from llg_emulator.metrics import bulk_rmse, correlation, rollout_metrics
from llg_emulator.model import with_mesh
from llg_emulator.plotting import plot_m_means
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import count_parameters

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tag", type=str, default="best")
    # a full rollout per trajectory is expensive; 16 is plenty for the curve
    parser.add_argument("--max-trajectories", type=int, default=16)
    # 3 x 101 .vtr frames + 3 .pvd time series, ~10 MB
    parser.add_argument("--vtr", action="store_true", help="write ParaView files")
    return parser.parse_args()


def rollout_correlation(model, paths):
    """Per-frame correlation, averaged over trajectories.

    One trajectory at a time and retargeted per trajectory, so a val set with
    mixed meshes evaluates fine even though it could not be batched.
    """
    corrs = []
    for path in tqdm(paths):
        trj = open_trajectory(path)
        m_true = jnp.asarray(_frames_to_cf(trj.m))
        m_pred = rollout_trajectory(
            with_mesh(model, trj.n, trj.dx),
            m_true,
            jnp.asarray(trj.H),
            s0=jnp.array([trj.s_enc(1)]),
        )
        corrs.append(correlation(pred=m_pred, ref=m_true)[0])
    corrs = jnp.stack(corrs, axis=0)
    return corrs.mean(axis=0), corrs.std(axis=0)


def sp4_trajectories(model):
    """Reference, predicted and error m(t) for SP4, all (t, 3, h, w)."""
    trj = open_trajectory(sp4_path())
    m_ref = jnp.asarray(_frames_to_cf(trj.m))
    m_pred = rollout_trajectory(
        with_mesh(model, trj.n, trj.dx),
        m_ref,
        jnp.asarray(trj.H),
        s0=jnp.array([trj.s_enc(1)]),
    )
    return trj, m_ref, m_pred


def write_sp4_vtr(model, out_dir: Path):
    """Write SP4 reference / predicted / error as ParaView time series.

    One `.vtr` per frame plus a `.pvd` collection carrying the physical time, so
    ParaView opens `sp4_<name>.pvd` and scrubs the 1 ns of dynamics directly.
    `error` is the vector `pred - ref`; ParaView derives its magnitude itself.

    `m` is stored on mesh *nodes* while `write_vtr` lays out cell data, so the
    nodal counts are handed over as the cell counts: every node renders as one
    voxel, offset half a cell from its true position. That is a rendering
    convention, not a resampling -- no values are interpolated.
    """
    trj, m_ref, m_pred = sp4_trajectories(model)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = (*m_ref.shape[2:], 1)  # (nodes_x, nodes_y, 1) treated as cell counts
    assert m_ref.shape[1] == 3 and len(n) == 3, m_ref.shape
    series = {
        "reference": np.asarray(m_ref),
        "predicted": np.asarray(m_pred),
        "error": np.asarray(m_pred - m_ref),
    }
    for name, traj in series.items():
        group = VtkGroup(str(out_dir / f"sp4_{name}"))
        for i, frame in enumerate(traj):
            stem = str(out_dir / f"sp4_{name}_{i:04d}")
            # (3, h, w) -> (h, w, 1, 3), the layout write_vtr indexes
            write_vtr(np.moveaxis(frame, 0, -1)[:, :, None, :], stem, n, trj.dx)
            group.addFile(stem + ".vtr", sim_time=i * trj.dt)
        group.save()

    err = np.linalg.norm(series["error"], axis=1)
    print(
        f"wrote {len(series)} x {len(m_ref)} .vtr frames + .pvd to {out_dir}/ "
        f"(grid {n[0]}x{n[1]}, max |error| {err.max():.4f}, mean {err.mean():.4f})"
    )


def run_benchmark(model, variant: str, out_dir: Path):
    """Roll out one benchmark trajectory and report bulk RMSE + correlation.

    Streamed frame by frame (`metrics.rollout_metrics`), so this holds regardless
    of mesh size — `large` is 2001x2001 nodal and its full history would not fit
    on the device.
    """
    path = benchmark_path(variant)
    trj = open_trajectory(path)
    ref, pred, corr = rollout_metrics(model, path)
    rmse = bulk_rmse(ref, pred)

    fig, _ = plot_m_means(m_avg=ref, m_avg_pred=pred)
    fig.savefig(out_dir / f"{variant}_bulk.png")
    plt.close(fig)

    print(
        f"{variant:>6}: mesh {str(trj.n):>12}  bulk_rmse {rmse:.4e}  "
        f"corr mean {corr.mean():.4f} final {corr[-1]:.4f}",
        flush=True,
    )
    return rmse, corr


def plot_rollout_correlation(cm, cs):
    xx = range(cm.shape[0])
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.plot(xx, cm)
    axs.fill_between(xx, cm + cs, cm - cs, alpha=0.4)
    axs.set_xlabel("step")
    axs.set_ylabel("correlation")
    return fig, axs


def main():
    args = _parse_args()
    model_path = Path(args.model_path)

    key = jr.PRNGKey(args.seed)
    key, subkey = jr.split(key)
    model = load_model(model_path, key=subkey, tag=args.tag)
    print(f"{count_parameters(model)} params, trained on mesh {model.demag.n}")

    # every benchmark variant: each on its own mesh, none seen during training
    for variant in BENCHMARK_VARIANTS:
        run_benchmark(model, variant, model_path)

    if args.vtr:
        write_sp4_vtr(model, model_path / "vtr")

    paths = sample_dirs(variant_dirs("val", exclude=BENCHMARK_VARIANTS))
    paths = paths[: args.max_trajectories]
    cm, cs = rollout_correlation(model, paths)
    fig, _ = plot_rollout_correlation(cm=cm, cs=cs)
    fig.savefig(model_path / "correlation.png")
    plt.close(fig)
    print(f"final-frame correlation over {len(paths)} val trajectories: {cm[-1]:.4f}")


if __name__ == "__main__":
    main()
