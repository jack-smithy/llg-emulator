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
    stride: int = 1,
    include_init: bool = True,
) -> Array:
    """Unroll the model from m_true[0], taking `stride`-sized steps.

    Each model call advances `stride` base steps (s_enc = log2(stride)); it makes
    (len-1)//stride calls with include_init (matching frames 0, stride, 2*stride…),
    or len//stride without. stride=1 reproduces the original one-step rollout.
    """
    assert H_ext.shape == (3,)
    s_enc = jnp.float32(jnp.log2(stride))
    steps = (m_true.shape[0] - 1) if include_init else m_true.shape[0]
    n = steps // stride
    return rollout(
        lambda x: model(x, H_ext, s_enc),
        n=n,
        include_init=include_init,
    )(m_true[0])


def rollout_trajectories(
    model: LLGEmulator,
    m_true,
    H_ext,
    *,
    stride: int = 1,
    include_init: bool = True,
):
    return jax.vmap(
        lambda m, h: rollout_trajectory(
            model, m, h, stride=stride, include_init=include_init
        )
    )(m_true, H_ext)
