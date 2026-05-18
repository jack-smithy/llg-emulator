import equinox as eqx
import jax
import jax.numpy as jnp
from einops import repeat
from jaxtyping import Array


def stepper_fn(m: Array, model: eqx.Module, H_ext: Array) -> Array:
    """One model step: concat m with the broadcast field and predict next m."""
    H_ext = repeat(H_ext, "c -> c h w", h=m.shape[1], w=m.shape[2])
    feature = jnp.concatenate((m, H_ext), axis=0)
    return model(feature)


def rollout(stepper_fn, n: int, *, include_init: bool = False):
    """Turn an autonomous timestepper into an efficient trajectory unroll."""

    def scan_fn(u, _):
        u_next = stepper_fn(u)
        return u_next, u_next

    def rollout_fn(init: Array) -> Array:
        _, history = jax.lax.scan(scan_fn, init, None, length=n)
        if include_init:
            return jnp.concatenate(
                [jnp.expand_dims(init, axis=0), history], axis=0
            )
        return history

    return rollout_fn


def rollout_trajectory(
    model: eqx.Module,
    m_true: Array,
    H_ext: Array,
    *,
    include_init: bool = True,
) -> Array:
    """Autoregressively roll out a trajectory matching m_true's length,
    starting from its initial state. Replaces the repeated
    rollout(lambda x: stepper_fn(x, model, H_ext), ...)(m_true[0]) pattern.
    """
    n = m_true.shape[0] - 1 if include_init else m_true.shape[0]
    return rollout(
        lambda x: stepper_fn(x, model, H_ext),
        n=n,
        include_init=include_init,
    )(m_true[0])
