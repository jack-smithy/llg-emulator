from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from pdequinox.arch import ClassicFNO


def spherical_to_cartesian(arr: Array) -> Array:
    theta, phi = arr[0], arr[1]
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack([x, y, z], axis=0)


class LLGEmulator(ClassicFNO):
    def __init__(
        self,
        hidden_channels: int = 32,
        num_modes: int = 12,
        num_blocks: int = 4,
        activation: Callable = jax.nn.gelu,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=2,
            in_channels=6,
            out_channels=2,
            hidden_channels=hidden_channels,
            num_modes=num_modes,
            num_blocks=num_blocks,
            activation=activation,
            boundary_mode=None,
            key=key,
        )

    def __call__(self, x) -> Array:
        x = super().__call__(x)
        x = spherical_to_cartesian(x)
        return x
