import jax.numpy as jnp
from jaxtyping import Array
from llg_emulator.rollout import rollout_trajectory, rollout_trajectories
from llg_emulator.data import load_trajectory
import grain
import numpy as np


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

    def loader(source, batch_size):
        num_trjs = len(source.fields)
        num_batches = num_trjs // batch_size

        for i in range(num_batches):
            yield (
                np.stack(source.trajs[i : i + batch_size]),
                np.stack(source.fields[i : i + batch_size]),
            )

    m_refs, m_preds = [], []
    for m_ref, field in loader(source, 2):
        m_pred = rollout_trajectories(model, m_ref, field)
        m_preds.append(m_pred)
        m_refs.append(m_ref)

    m_ref = jnp.concat(m_refs, axis=0)
    m_pred = jnp.concat(m_preds, axis=0)

    mean, std = correlation(m_pred, m_ref)
    return mean, std


def bulk_magnetization(model, path):
    m_true, H_ext = load_trajectory(path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    return jnp.mean(m_true, axis=(2, 3)), jnp.mean(m_pred, axis=(2, 3))
