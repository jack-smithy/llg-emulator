from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
from pdequinox.arch import ClassicFNO


def spherical_to_cartesian(arr: Array) -> Array:
    theta, phi = arr[0], arr[1]
    x = jnp.sin(theta) * jnp.cos(phi)
    y = jnp.sin(theta) * jnp.sin(phi)
    z = jnp.cos(theta)
    return jnp.stack([x, y, z], axis=0)


class _FiLM(eqx.Module):
    """Maps the H_ext 3-vector to per-(stage, channel) (gamma, beta).

    Final layer is zero-initialised so at start gamma=1, beta=0 (identity
    modulation) and the model reduces to the plain residual path.
    """

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, in_dim: int, hidden: int, out_dim: int, *, key):
        k1, k2 = jr.split(key)
        self.l1 = eqx.nn.Linear(in_dim, hidden, key=k1)
        l2 = eqx.nn.Linear(hidden, out_dim, key=k2)
        self.l2 = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            l2,
            (jnp.zeros_like(l2.weight), jnp.zeros_like(l2.bias)),
        )

    def __call__(self, h: Array) -> Array:
        return self.l2(jax.nn.gelu(self.l1(h)))


class LLGEmulator(ClassicFNO):
    """FNO predicting next m via a FiLM-conditioned tangent residual.

    The 6-channel input feature [m_t (3), H_ext broadcast (3)] is split
    inside the model: only m_t flows through the FNO (in_channels=3), while
    the constant H_ext vector drives a FiLM conditioner that modulates the
    hidden features after the lifting layer and after every block. The FNO
    output dm is projected onto m_t's tangent plane (LLG keeps |m|=1), then
    m_{t+1} = normalize(m_t + dm_perp).
    """

    film: _FiLM
    n_pts: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_channels: int = 32,
        num_modes: int = 12,
        num_blocks: int = 4,
        activation: Callable = jax.nn.gelu,
        *,
        key: PRNGKeyArray,
    ):
        fno_key, film_key = jr.split(key)
        super().__init__(
            num_spatial_dims=2,
            in_channels=3,
            out_channels=3,
            hidden_channels=hidden_channels,
            num_modes=num_modes,
            num_blocks=num_blocks,
            activation=activation,
            boundary_mode=None,
            key=fno_key,
        )
        self.n_pts = num_blocks + 1  # post-lifting + one per block
        self.film = _FiLM(
            3, hidden_channels, self.n_pts * 2 * hidden_channels, key=film_key
        )

    def __call__(self, x: Array) -> Array:
        m = x[:3]  # m_t
        H = x[3:, 0, 0]  # constant H_ext 3-vector (broadcast in space)

        g = self.film(H).reshape(self.n_pts, 2, -1)
        # gamma = 1 + raw so zero-init -> identity modulation
        g = g.at[:, 0, :].add(1.0)

        def modulate(h, i):
            return g[i, 0][:, None, None] * h + g[i, 1][:, None, None]

        h = modulate(self.lifting(m), 0)
        for i, block in enumerate(self.blocks, start=1):
            h = modulate(block(h), i)
        dm = self.projection(h)

        # tangent-space residual: the true change is perpendicular to m
        m_hat = m / (jnp.linalg.norm(m, axis=0, keepdims=True) + 1e-8)
        dm = dm - jnp.sum(dm * m_hat, axis=0, keepdims=True) * m_hat

        out = m + dm
        return out / (jnp.linalg.norm(out, axis=0, keepdims=True) + 1e-8)
