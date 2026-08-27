import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from llg_emulator.data import _frame_to_cf, open_trajectory
from llg_emulator.model import with_mesh


def nRMSE(pred, ref) -> Array:
    return jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)


def correlation(pred, ref):
    """Per-frame mean cosine similarity between predicted and reference spins.

    Both fields are unit vectors, so the pointwise dot product is cos(angle) and
    the frame score is its spatial mean: 1.0 is exact, 0.0 is orthogonal, -1.0 is
    inverted. This used to RMS the dot product, which scored a *perfectly
    inverted* field 1.0 -- the one failure mode the metric most needs to catch.
    """
    assert pred.shape == ref.shape
    assert len(pred.shape) >= 4
    # both shape (..., t, c, nx, ny)
    pointwise_corr = jnp.sum(pred * ref, axis=-3)  # (..., t, nx, ny)
    framewise_corr = jnp.mean(pointwise_corr, axis=(-1, -2))  # (..., t,)
    if len(framewise_corr.shape) == 2:  # if batch axis, reduce over batches
        return framewise_corr.mean(axis=0), framewise_corr.std(axis=0)
    return framewise_corr, jnp.zeros_like(framewise_corr)


def MSE(pred: Array, ref: Array) -> Array:
    return jnp.mean(jnp.square(pred - ref))


def rollout_metrics(model, path):
    """Stream an autoregressive rollout over one trajectory.

    Returns `(bulk_ref, bulk_pred, corr)` — the reference and predicted ⟨m⟩(t),
    each `(t, 3)`, and the per-frame mean cosine similarity `(t,)`.

    O(1) in trajectory length: one predicted frame and one reference frame are
    live at a time. The whole history does not fit for every benchmark —
    `data/val/large` is 2001x2001 nodal, 4.85 GB on disk, so prediction plus
    reference would be ~10 GB of device memory to produce a (t, 3) curve.

    Mesh-agnostic: the model is retargeted at the trajectory's own grid and the
    step size comes from the trajectory's own `dt`.
    """
    trj = open_trajectory(path)
    model = with_mesh(eqx.nn.inference_mode(model), trj.n, trj.dx)
    cond = jnp.concatenate((jnp.asarray(trj.H), jnp.array([trj.s_enc(1)])))
    step = eqx.filter_jit(lambda m: model(m, cond))

    m = jnp.asarray(_frame_to_cf(trj.m, 0))
    bulk_ref, bulk_pred, corr = [], [], []
    for t in range(trj.m.shape[0]):
        if t:
            m = step(m)
        ref = jnp.asarray(_frame_to_cf(trj.m, t))
        bulk_ref.append(np.asarray(ref.mean(axis=(1, 2))))
        bulk_pred.append(np.asarray(m.mean(axis=(1, 2))))
        corr.append(float(jnp.mean(jnp.sum(m * ref, axis=0))))
    return np.array(bulk_ref), np.array(bulk_pred), np.array(corr)


def bulk_magnetization(model, path):
    """Reference vs. rolled-out bulk magnetization ⟨m⟩(t) for one trajectory."""
    bulk_ref, bulk_pred, _ = rollout_metrics(model, path)
    return bulk_ref, bulk_pred


def bulk_rmse(ref, pred) -> float:
    """RMSE between two ⟨m⟩(t) curves — the SP4 headline metric."""
    return float(jnp.sqrt(jnp.mean((jnp.asarray(ref) - jnp.asarray(pred)) ** 2)))
