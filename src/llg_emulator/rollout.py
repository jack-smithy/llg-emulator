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


s0_default = jnp.array([0.0])


def rollout_trajectory(
    model: LLGEmulator,
    m_true,
    H_ext,
    s0=jnp.array([0.0]),
    *,
    stride: int = 1,
    include_init: bool = True,
) -> Array:
    """Unroll the model from m_true[0], one model call per `stride` reference frames.

    `s0` is the step-size encoding the model is conditioned on and `stride` is how
    many frames of `m_true` one call covers; the caller owns both because only it
    knows the trajectory's `dt` (see `data.Trajectory.s_enc`). They have to agree:
    `s0 = log2(stride * dt / BASE_STEP_TIME)`.

    Makes `(len - 1) // stride` calls with `include_init` (lining up with frames
    0, stride, 2*stride, ...), or `len // stride` without.
    """
    assert H_ext.shape == (3,)
    cond = jnp.concat((H_ext, s0))

    n = m_true.shape[0] - 1 if include_init else m_true.shape[0]
    return rollout(
        lambda x: model(x, cond),
        n=n // stride,
        include_init=include_init,
    )(m_true[0])


def rollout_trajectories(
    model: LLGEmulator,
    m_true,
    H_ext,
    *,
    include_init: bool = True,
):
    return jax.vmap(
        lambda m, h: rollout_trajectory(model, m, h, include_init=include_init)
    )(m_true, H_ext)
