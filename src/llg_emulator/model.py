from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray
from pdequinox.arch import ClassicResNet

from llg_emulator.physics import DemagField, demag_for


class FiLM(eqx.Module):
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
    # mesh *cell* counts (2D thin film); the nodal fields the solver writes -
    # and the model consumes - are (mesh_n[0] + 1, mesh_n[1] + 1). See physics.py.
    mesh_n: tuple = (256, 256)
    mesh_dx: tuple = (5e-9, 5e-9, 3e-9)
    demag_p: int = 20
    # FiLM conditioning: [H_ext / Ms (3), s_enc (1)]. Every call site appends
    # s_enc unconditionally, so 4 is the only width the code can actually run.
    cond_dim: int = 4


class LLGEmulator(eqx.Module):
    film: FiLM
    backbone: ClassicResNet
    demag: DemagField
    n_pts: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        demag: DemagField,
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
        # Zero the projection so `dm` starts at 0 and the model *is* the identity
        # map at step 0 -- the target m_{t+1} is within ~1e-2 of m_t, so a random
        # projection starts the run far from the answer. Mirrors the FiLM zero-init
        # above: without both, that one is wasted.
        params, rest = eqx.partition(self.backbone.projection, eqx.is_inexact_array)
        self.backbone = eqx.tree_at(
            lambda b: b.projection,
            self.backbone,
            eqx.combine(jtu.tree_map(jnp.zeros_like, params), rest),
        )
        self.n_pts = config.num_blocks + 1  # post-lifting + one per block
        # FiLM conditions on H_ext (3) plus, when cond_dim==4, a log step-size
        # scalar so the model can take variable-Δt steps (see `cond` below).
        self.cond_dim = config.cond_dim
        self.film = FiLM(
            in_features=config.cond_dim,
            hidden_channels=config.hidden_channels,
            out_features=self.n_pts * 2 * config.hidden_channels,
            key=film_key,
        )
        # Ms cancels under the nondim (h_demag / Ms) output, so any positive
        # value gives the nondimensionalised demag field the model consumes.
        self.demag = demag

    def step(self, m0, dm):
        # tangent-space residual: the true change is perpendicular to m
        m_hat = m0 / (jnp.linalg.norm(m0, axis=0, keepdims=True) + 1e-8)
        dm = dm - jnp.sum(dm * m_hat, axis=0, keepdims=True) * m_hat
        m1 = m0 + dm
        return m1 / (jnp.linalg.norm(m1, axis=0, keepdims=True) + 1e-8)

    def init_film(self, cond):
        g = self.film(cond).reshape(self.n_pts, 2, -1)
        g = g.at[:, 0, :].add(1.0)

        def modulate(h, i):
            return g[i, 0][:, None, None] * h + g[i, 1][:, None, None]

        return modulate

    def forward(self, m0, cond):
        modulate = self.init_film(cond)

        model_in = jnp.concatenate([m0, self.demag(m0)], axis=0)  # (6, nx, ny)
        h = modulate(self.backbone.lifting(model_in), 0)  # type: ignore
        for i, block in enumerate(self.backbone.blocks, start=1):
            h = modulate(block(h), i)  # type: ignore
        dm = self.backbone.projection(h)  # type: ignore

        return dm

    def __call__(self, m0: Array, cond: Array) -> Array:
        """Predict m one step of size `2**s_enc * 10 ps` ahead, given
        `cond = [H_ext / Ms (3), s_enc (1)]` (s_enc = 0 -> one base step).

        Mesh-agnostic in the spatial dims: the backbone is fully convolutional
        and `self.demag` supplies the only mesh-shaped operator. Use `with_mesh`
        to retarget a trained model at a different grid."""
        dm = self.forward(m0=m0, cond=cond)
        m1 = self.step(m0=m0, dm=dm)
        return m1


def with_mesh(model: LLGEmulator, n, dx) -> LLGEmulator:
    """Same weights, demag rebuilt for the mesh `(n, dx)`.

    The backbone is fully convolutional, so it transfers to any grid unchanged;
    the demag tensor is the one mesh-shaped part and has to be rebuilt. `dx` must
    match the mesh the weights were trained on -- a different cell size changes
    the physics per cell, which no amount of rebuilding fixes.
    """
    n, dx = tuple(int(x) for x in n), tuple(float(x) for x in dx)
    if dx != model.demag.dx:
        raise ValueError(
            f"cell size {dx} != the model's {model.demag.dx}; the learned step map "
            f"is only valid at the cell size it was trained on."
        )
    if n == model.demag.n:
        return model
    demag = demag_for(n, dx, model.demag.Ms, model.demag.p)
    return eqx.tree_at(lambda m: m.demag, model, demag)
