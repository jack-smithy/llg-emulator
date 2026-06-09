import jax
import jax.numpy as jnp
from jaxtyping import Array

from llg_emulator.model import LLGEmulator


def rollout(stepper_fn, n: int, *, include_init: bool = False):
    """Turn an autonomous timestepper into an efficient trajectory unroll."""

    def scan_fn(u, _):
        u_next = stepper_fn(u)
        return u_next, u_next

    def rollout_fn(init: Array) -> Array:
        _, history = jax.lax.scan(scan_fn, init, None, length=n)
        if include_init:
            return jnp.concatenate([jnp.expand_dims(init, axis=0), history], axis=0)
        return history

    return rollout_fn


def rollout_trajectory(
    model: LLGEmulator,
    m_true,
    H_ext,
    *,
    include_init: bool = True,
) -> Array:
    assert H_ext.shape == (3,)
    n = m_true.shape[0] - 1 if include_init else m_true.shape[0]
    return rollout(
        lambda x: model(x, H_ext),
        n=n,
        include_init=include_init,
    )(m_true[0])
