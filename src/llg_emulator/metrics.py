import jax.numpy as jnp
from jaxtyping import Array
from llg_emulator.rollout import rollout_trajectories


def nRMSE(pred, ref) -> Array:
    return jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)


def correlation(pred, ref):
    assert pred.shape == ref.shape
    assert len(pred.shape) >= 4
    # both shape (..., t, c, nx, ny)
    pointwise_corr = jnp.sum(pred * ref, axis=-3)  # (..., t, nx, ny)
    framewise_corr = jnp.sqrt(jnp.mean(pointwise_corr**2, axis=(-1, -2)))  # (..., t,)
    return framewise_corr.mean()  # ()


def MSE(pred: Array, ref: Array) -> Array:
    return jnp.mean(jnp.square(pred - ref))


def bulk_magnetization(model, source):
    m_ref = jnp.stack(source.trajs, axis=0)
    h_ext = jnp.stack(source.fields, axis=0)
    m_pred = rollout_trajectories(m_ref, h_ext, model)
    return jnp.mean(m_ref, axis=(-1, -2)), jnp.mean(m_pred, axis=(-1, -2))
