import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.data import load_trajectory


def nRMSE(pred, ref) -> Array:
    return jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)


def correlation(pred, ref):
    assert pred.shape == ref.shape
    assert len(pred.shape) >= 4
    # both shape (..., t, c, nx, ny)
    pointwise_corr = jnp.sum(pred * ref, axis=-3)  # (..., t, nx, ny)
    framewise_corr = jnp.sqrt(jnp.mean(pointwise_corr**2, axis=(-1, -2)))  # (..., t,)
    if len(framewise_corr.shape) == 2:  # if batch axis, reduce over batches
        return framewise_corr.mean(axis=0), framewise_corr.std(0)
    return framewise_corr, jnp.zeros_like(framewise_corr)


def MSE(pred: Array, ref: Array) -> Array:
    return jnp.mean(jnp.square(pred - ref))


def correlation_epoch(model, source):
    """Framewise field-correlation vs step, averaged over the source's trajectories.

    Streams one trajectory at a time (rollout + spatial reduction) so memory stays
    O(one trajectory), not O(whole val set) — required for large datasets. Returns
    (mean, std) over trajectories, each shape (t,).
    """
    store = source.store
    per_traj = []
    for ti in range(len(store)):
        m_ref = jnp.asarray(store.full(ti))  # (t, 3, nx, ny)
        H = jnp.asarray(store.fields[ti])
        m_pred = rollout_trajectory(model, m_ref, H, stride=1, include_init=True)
        fc, _ = correlation(m_pred, m_ref)  # single-traj -> framewise (t,)
        per_traj.append(np.asarray(fc))
    stacked = jnp.asarray(np.stack(per_traj, axis=0))  # (n_traj, t) — already reduced
    return stacked.mean(axis=0), stacked.std(axis=0)


def bulk_magnetization(model, path, stride: int = 1):
    m_true, H_ext, _dt = load_trajectory(path)
    m_pred = rollout_trajectory(model, m_true, H_ext, stride=stride, include_init=True)
    ref = m_true[0 : m_pred.shape[0] * stride : stride]
    return jnp.mean(ref, axis=(2, 3)), jnp.mean(m_pred, axis=(2, 3))
