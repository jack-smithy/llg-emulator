from pathlib import Path
import jax.numpy as jnp
from einops import rearrange, repeat
import json
import jax


def load_trajectory(path: Path):
    m = jnp.load(path / "m.npy")
    m = m[1:, :, :, 0, :]
    m = rearrange(m, "t w h c -> t c w h")

    with open(path / "params.json", "r") as f:
        p = json.load(f)

    H_ext = jnp.asarray(p["H_ext"])
    Ms = p["material"]["Ms"]
    H_ext_nondimensionalized = H_ext / Ms

    return m, H_ext_nondimensionalized


def stepper_fn(m, model, H_ext):
    H_ext = repeat(H_ext, "c -> c h w", h=m.shape[1], w=m.shape[2])
    feature = jnp.concatenate((m, H_ext), axis=0)
    return model(feature)


def rollout(
    stepper_fn,
    n,
    *,
    include_init=False,
):
    """
    Transform an autonomous timestepper into a function that efficiently unrolls
    a trajectory.
    """

    def scan_fn(u, _):
        u_next = stepper_fn(u)
        return u_next, u_next

    def rollout_fn(init):
        _, history = jax.lax.scan(
            scan_fn,
            init,
            None,
            length=n,
        )

        if include_init:
            return jnp.concatenate(
                [
                    jnp.expand_dims(init, axis=0),
                    history,
                ],
                axis=0,
            )
        else:
            return history

    return rollout_fn
