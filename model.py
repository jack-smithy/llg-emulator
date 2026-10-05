"""Emulators of one LLG frame (10 ps) on a film's mesh.

`ResidualEmulator` is the plain network, m' = normalise(m + N(inputs)); its
solver-in-the-loop and demag-input variants are arranged by `utils.model_step`.

`ClosureEmulator` (`ModelConfig.closure`) is the learned-closure LLG: the coarse LLG
solver (`physics.LLGStepper`: magnum.np's exchange, the exact demag of the coarse
mesh, the applied field) stepped under one more effective-field term, a closure the
network reads off the coarse state,

    m' = LLG_dt[m; H_ext + H_theta(m)]
    H_theta(m) = eps Ms N(m, h_ex / Ms, h_d / Ms; H_ext / Ms, eps),   eps = l_ex / delta

- Structure: any tangential velocity of a unit vector is m x H for some H, so a field
  closure gives up nothing against a correction of m, while the step stays a damped
  precession: |m| = 1 exactly, and a bounded closure gives a bounded one-step map, so
  rollouts cannot run away faster than the LLG itself (the plain networks' do). At zero
  closure the model *is* coarse micromagnetics (`zero_init`).
- Scale: with material and thickness fixed, the one discretisation parameter of the
  dimensionless problem is eps = l_ex / delta. The network sees only fields in units
  of Ms on the coarse mesh -- m, the coarse exchange field h_ex = eps^2 L[m] (which
  carries a finite-difference Laplacian's 1 / delta^2), the exact demag field h_d, the
  applied field -- and eps itself. Coarsening the mesh drives eps and h_ex towards
  zero, i.e. into the interior edge of the training range, where a log(delta / l_ex)
  input would run off its end.
- Leading order: a sub-cell closure is a surface effect (a wall inside a cell of size
  delta moves the cell mean at a rate ~ v / delta), so it scales as eps times a
  delta-free function of the texture. The eps prefactor makes the network's job
  delta-free at leading order and sends the closure to zero -- the model back to
  coarse micromagnetics -- as delta grows; higher orders come through the eps input.
- Locality: the network is a compact stencil (ClassicResNet, kernel 3, no
  normalisation), as a sub-grid closure should be; the long-range physics (demag) is
  exact. Film size and shape transfer, and the network tiles exactly
  (`receptive_radius`).
"""

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as LA
import pdequinox as pdeqx
from pdequinox.arch import ClassicResNet, DilatedResNet

from physics import exchange_field

COND_DIM = 3  # [Hx, Hy] / Ms, log(delta / l_ex)
ARCHS = {"dilated": DilatedResNet, "classic": ClassicResNet}


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_blocks: int = 2
    # "dilated": DilatedResNet (dilations 1..8, 22 cells per block); "classic":
    # ClassicResNet (two kernel-3 convolutions per block, 2 cells per block)
    arch: str = "dilated"
    # solver-in-the-loop (Um et al. 2020): the network corrects the coarse solver's
    # step, m' = P(m) + C(P(m)), and sees only P(m); otherwise it maps m + coords
    solver_in_the_loop: bool = False
    # the latest frame's exact demag field (physics.demag_cache) as 3 more inputs
    use_demag: bool = False
    # frames of context: the latest m plus the in_frames - 1 before it
    in_frames: int = 1
    # the learned-closure LLG (`ClosureEmulator`, module docstring)
    closure: bool = False
    # zero the output projection at init, so the untrained model adds exactly nothing
    # (for solver-in-the-loop and the closure: starts as the coarse solver itself)
    zero_init: bool = False
    activation: Callable = jax.nn.gelu


class ResidualEmulator(eqx.Module):
    network: eqx.Module

    def __init__(self, network, normalization_factor: float = 1.0):
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(
            network=network, normalization_factor=normalization_factor
        )

    def __call__(self, m0, meta_data):
        dm = self.network(m0, meta_data=meta_data)  # type: ignore
        m1 = m0[:3] + dm
        return m1 / LA.norm(m1, axis=0, keepdims=True)


class ClosureEmulator(eqx.Module):
    """The learned-closure LLG (module docstring). One sample at a time: m (3, nx, ny)
    unit vectors on the cells, cond (3,) as `utils.conditioning` returns it, h (3,)
    the applied field in A/m, solver the mesh's `physics.LLGStepper`."""

    network: eqx.Module

    def __init__(self, network, normalization_factor: float = 1.0):
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(
            network=network, normalization_factor=normalization_factor
        )

    @staticmethod
    def eps(cond):
        """l_ex / delta; `utils.conditioning` records log(delta / l_ex)."""
        return jnp.exp(-cond[2])

    def features(self, m, cond, solver):
        """The network's inputs: fields (9, nx, ny) `[m, h_ex / Ms, h_d / Ms]` and the
        constant channels (3,) `[Hx / Ms, Hy / Ms, eps]`."""
        Ms = solver.Ms
        h_ex = exchange_field(jnp.moveaxis(m, 0, -1), solver.dx, Ms, solver.A)
        x = jnp.concatenate([m, jnp.moveaxis(h_ex, -1, 0) / Ms, solver.demag(m) / Ms])
        return x, jnp.stack([cond[0], cond[1], self.eps(cond)])

    def field(self, m, cond, solver):
        """H_theta (3, nx, ny) in A/m."""
        x, meta = self.features(m, cond, solver)
        return self.eps(cond) * solver.Ms * self.network(x, meta_data=meta)  # type: ignore

    def __call__(self, m, cond, h, solver):
        return solver(m, h[:, None, None] + self.field(m, cond, solver))


def in_channels(config: ModelConfig) -> int:
    """P(m) (3) for solver-in-the-loop; m + h_ex + h_d (9) for the closure; otherwise
    the in_frames frames (3 each) + coords (2) [+ demag (3)]; plus the embedded
    conditioning."""
    if config.closure:
        if config.solver_in_the_loop or config.use_demag or config.in_frames != 1:
            raise ValueError("the closure model takes one frame and builds its inputs")
        return 9 + COND_DIM
    if config.solver_in_the_loop:
        if config.use_demag or config.in_frames != 1:
            raise ValueError("solver-in-the-loop sees P(m) alone: no demag, one frame")
        return 3 + COND_DIM
    return 3 * config.in_frames + 2 + 3 * config.use_demag + COND_DIM


def build_model(config: ModelConfig, key) -> eqx.Module:
    # fixed receptive field in cells, so the weights transfer to any film size;
    # no GroupNorm, which normalises over the whole film and makes the net non-local
    network = ARCHS[config.arch](
        num_spatial_dims=2,
        in_channels=in_channels(config),
        out_channels=3,
        hidden_channels=config.hidden_channels,
        num_blocks=config.num_blocks,
        activation=config.activation,
        boundary_mode="neumann",  # reflect padding: finite thin film, not periodic
        use_norm=False,
        key=key,
    )
    if config.zero_init:
        network = eqx.tree_at(
            lambda n: (n.projection.weight, n.projection.bias),
            network,
            replace_fn=jnp.zeros_like,
        )
    if config.closure:
        return ClosureEmulator(network=network)
    return ResidualEmulator(network=network)


def receptive_radius(model: eqx.Module) -> int:
    """Cells the network reads on each side of a cell: the halo that makes tiled
    inference of this strictly local network exact."""
    return math.ceil(max(max(r) for r in model.network.network.receptive_field))


def save_model(model: eqx.Module, config: ModelConfig, path: Path, tag: str):
    """Serialise the weights to `<path>/<tag>.eqx` + `metadata.json`."""
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / f"{tag}.eqx", model)
    meta = {
        "hidden_channels": config.hidden_channels,
        "num_blocks": config.num_blocks,
        "arch": config.arch,
        "solver_in_the_loop": config.solver_in_the_loop,
        "use_demag": config.use_demag,
        "in_frames": config.in_frames,
        "closure": config.closure,
        "zero_init": config.zero_init,
    }
    # rewrite every time: a stale sidecar builds the wrong skeleton on load
    (path / "metadata.json").write_text(json.dumps(meta))


def load_config(path: Path) -> ModelConfig:
    """The `ModelConfig` a run's `metadata.json` records."""
    meta = json.loads((path / "metadata.json").read_text())
    return ModelConfig(
        hidden_channels=meta["hidden_channels"],
        num_blocks=meta["num_blocks"],
        arch=meta.get("arch", "dilated"),
        solver_in_the_loop=meta.get("solver_in_the_loop", False),
        use_demag=meta.get("use_demag", False),
        in_frames=meta.get("in_frames", 1),
        closure=meta.get("closure", False),
        zero_init=meta.get("zero_init", False),
    )


def load_model(path: Path, key, tag: str) -> eqx.Module:
    """Rebuild the model from `<tag>.eqx` + `metadata.json`. `key` only seeds
    the skeleton whose weights are then overwritten."""
    model = build_model(load_config(path), key)
    return eqx.tree_deserialise_leaves(path / f"{tag}.eqx", model)
