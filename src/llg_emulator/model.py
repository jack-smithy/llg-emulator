from typing import Callable

from dataclasses import dataclass
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
from pdequinox.arch import ClassicResNet

from llg_emulator.physics import DemagField


class FiLM(eqx.Module):
    """Maps the H_ext 3-vector to per-(stage, channel) (gamma, beta).

    Final layer is zero-initialised so at start gamma=1, beta=0 (identity
    modulation) and the model reduces to the plain residual path.
    """

    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        hidden_channels: int,
        out_features: int,
        *,
        key,
    ):
        k1, k2 = jr.split(key)
        self.l1 = eqx.nn.Linear(in_features, hidden_channels, key=k1)
        l2 = eqx.nn.Linear(hidden_channels, out_features, key=k2)

        assert l2.bias is not None

        self.l2 = eqx.tree_at(
            lambda m: (m.weight, m.bias),
            l2,
            (jnp.zeros_like(l2.weight), jnp.zeros_like(l2.bias)),
        )

    def __call__(self, h: Array) -> Array:
        return self.l2(jax.nn.gelu(self.l1(h)))


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_blocks: int = 4
    activation: Callable = jax.nn.gelu
    mesh_n: tuple = (256, 256, 1)
    mesh_dx: tuple = (5e-9, 5e-9, 3e-9)
    demag_p: int = 20


class LLGEmulator(eqx.Module):
    film: FiLM
    backbone: ClassicResNet
    demag: DemagField
    n_pts: int = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        model_key, film_key = jr.split(key)
        self.backbone = ClassicResNet(
            num_spatial_dims=2,
            in_channels=6,  # m_t (3) + demag(m_t) (3)
            out_channels=3,
            hidden_channels=config.hidden_channels,
            num_blocks=config.num_blocks,
            activation=config.activation,
            boundary_mode="neumann",
            key=model_key,
        )
        self.n_pts = config.num_blocks + 1  # post-lifting + one per block
        self.film = FiLM(
            in_features=3,
            hidden_channels=config.hidden_channels,
            out_features=self.n_pts * 2 * config.hidden_channels,
            key=film_key,
        )
        # Ms cancels under the nondim (h_demag / Ms) output, so any positive
        # value gives the nondimensionalised demag field the model consumes.
        self.demag = DemagField(config.mesh_n, config.mesh_dx, Ms=1.0, p=config.demag_p)

    def step(self, m0, dm):
        # tangent-space residual: the true change is perpendicular to m
        m_hat = m0 / (jnp.linalg.norm(m0, axis=0, keepdims=True) + 1e-8)
        dm = dm - jnp.sum(dm * m_hat, axis=0, keepdims=True) * m_hat
        m1 = m0 + dm
        return m1 / (jnp.linalg.norm(m1, axis=0, keepdims=True) + 1e-8)

    def init_film(self, H):
        g = self.film(H).reshape(self.n_pts, 2, -1)
        g = g.at[:, 0, :].add(1.0)

        def modulate(h, i):
            return g[i, 0][:, None, None] * h + g[i, 1][:, None, None]

        return modulate

    def forward(self, m0, H):
        modulate = self.init_film(H)

        model_in = jnp.concatenate([m0, self.demag(m0)], axis=0)  # (6, nx, ny)
        h = modulate(self.backbone.lifting(model_in), 0)  # type: ignore
        for i, block in enumerate(self.backbone.blocks, start=1):
            h = modulate(block(h), i)  # type: ignore
        dm = self.backbone.projection(h)  # type: ignore

        return dm

    def __call__(self, m0: Array, H: Array) -> Array:
        dm = self.forward(m0=m0, H=H)
        m1 = self.step(m0=m0, dm=dm)
        return m1
