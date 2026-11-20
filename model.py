"""Two ways for a network to correct coarse micromagnetics, the LLG stepper P on the coarse
mesh (exchange, its exact demag field, the applied field):

solver-in-the-loop (Um et al. 2020, `SolverInTheLoop`, the default): the network corrects
the coarse step's output,

    m' = normalise(P(m) + N(P(m), h_ex / Ms, h_d / Ms; H_ext / Ms, eps)),   eps = l_ex / delta

the learned closure (`ClosureEmulator`): the coarse LLG stepped under one extra field,

    m' = LLG_dt[m; H_ext + H_theta(m)]
    H_theta(m) = eps Ms N(m, h_ex / Ms, h_d / Ms; H_ext / Ms, eps)

the energy closure (`EnergyClosure`): the closure field as the gradient of a learned local
energy of m alone (units of mu0 Ms^2 per cell volume), so every relaxation lowers an energy

    H_theta(m) = -eps Ms d/dm sum_cells e_theta(m; eps)

and its two-parameter limit (`ExchangeRescale`), the coarse-grained domain solver's idea of a
renormalised exchange constant, A_eff / A = exp(a0 + a1 log eps).

The networks see fields in units of Ms on the coarse mesh, plus eps. On the same recipe
solver-in-the-loop is the more accurate at 5-40 nm cells; the closure's eps prefactor sends
it back to the plain solver on much coarser, untrained meshes (160-320 nm), and a large
closure field can make the LLG too stiff for the integrator.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pdequinox as pdeqx
import yaml
from jaxtyping import Array, Float, PRNGKeyArray
from pdequinox.arch import ClassicResNet, DilatedResNet

from llg.constants import MU_0
from llg.physics import LLGStepper, exchange_field

ARCHS = {"classic": ClassicResNet, "dilated": DilatedResNet}


@dataclass
class ModelConfig:
    arch: str = "classic"
    hidden_channels: int = 64
    num_blocks: int = 4
    zero_init: bool = True  # start as the coarse solver itself
    formulation: str = "sil"  # solver-in-the-loop, "closure", "energy" or "aeff"
    fields: bool = True  # read h_ex and h_d besides m; without, the original SIL's inputs


def cell_eps(solver: LLGStepper) -> Float[Array, ""]:
    """l_ex / dx of the solver's mesh."""
    return jnp.sqrt(2 * solver.A / MU_0) / solver.Ms / solver.dx[0]


def network_inputs(
    m: Float[Array, "..."], h: Float[Array, "..."], solver: LLGStepper, fields: bool
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Channels-first m [, h_ex, h_d] / Ms on the coarse mesh, and [Hx / Ms, Hy / Ms, eps]."""
    Ms = solver.Ms
    eps = cell_eps(solver)
    x = [m]
    if fields:
        x += [exchange_field(m, solver.dx, Ms, solver.A) / Ms, solver.demag(m) / Ms]
    x = jnp.moveaxis(jnp.concatenate(x, axis=-1), -1, 0)
    return x, jnp.stack([h[0] / Ms, h[1] / Ms, eps])


class ClosureEmulator(eqx.Module):
    network: pdeqx.ConstantEmbeddingMetadataNetwork
    fields: bool = eqx.field(static=True)

    def __init__(self, network: eqx.Module, fields: bool = True):
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(network, 1.0)
        self.fields = fields

    def closure_field(
        self, m: Float[Array, "..."], h: Float[Array, "..."], solver: LLGStepper
    ) -> Float[Array, "..."]:
        x, scalars = network_inputs(m, h, solver, self.fields)
        out = self.network(x, meta_data=scalars)
        return scalars[2] * solver.Ms * jnp.moveaxis(out, 0, -1)

    def __call__(
        self, m: Float[Array, "..."], h: Float[Array, "..."], solver: LLGStepper
    ) -> Float[Array, "..."]:
        return solver(m, h + self.closure_field(m, h, solver))


class SolverInTheLoop(eqx.Module):
    network: pdeqx.ConstantEmbeddingMetadataNetwork
    fields: bool = eqx.field(static=True)

    def __init__(self, network: eqx.Module, fields: bool = True):
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(network, 1.0)
        self.fields = fields

    def __call__(
        self, m: Float[Array, "..."], h: Float[Array, "..."], solver: LLGStepper
    ) -> Float[Array, "..."]:
        p = solver(m, h)
        x, scalars = network_inputs(p, h, solver, self.fields)
        m = p + jnp.moveaxis(self.network(x, meta_data=scalars), 0, -1)
        return m / jnp.linalg.norm(m, axis=-1, keepdims=True)


class EnergyClosure(eqx.Module):
    network: pdeqx.ConstantEmbeddingMetadataNetwork

    def __init__(self, network: eqx.Module, fields: bool = False):
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(network, 1.0)

    def energy(self, m: Float[Array, "nx ny 3"], eps: Float[Array, ""]) -> Float[Array, ""]:
        return self.network(jnp.moveaxis(m, -1, 0), meta_data=eps[None]).sum()

    def closure_field(
        self, m: Float[Array, "nx ny 3"], h: Float[Array, "3"], solver: LLGStepper
    ) -> Float[Array, "nx ny 3"]:
        eps = cell_eps(solver)
        return -eps * solver.Ms * jax.grad(self.energy)(m, eps)

    def __call__(
        self, m: Float[Array, "nx ny 3"], h: Float[Array, "3"], solver: LLGStepper
    ) -> Float[Array, "nx ny 3"]:
        return solver(m, h + self.closure_field(m, h, solver))


class ExchangeRescale(eqx.Module):
    log_scale: Float[Array, "2"]  # a0, a1

    def __init__(self):
        self.log_scale = jnp.zeros(2)

    def closure_field(
        self, m: Float[Array, "nx ny 3"], h: Float[Array, "3"], solver: LLGStepper
    ) -> Float[Array, "nx ny 3"]:
        scale = jnp.exp(self.log_scale[0] + self.log_scale[1] * jnp.log(cell_eps(solver)))
        return (scale - 1) * exchange_field(m, solver.dx, solver.Ms, solver.A)

    def __call__(
        self, m: Float[Array, "nx ny 3"], h: Float[Array, "3"], solver: LLGStepper
    ) -> Float[Array, "nx ny 3"]:
        return solver(m, h + self.closure_field(m, h, solver))


Emulator = ClosureEmulator | SolverInTheLoop | EnergyClosure | ExchangeRescale
FORMULATIONS = {"closure": ClosureEmulator, "sil": SolverInTheLoop, "energy": EnergyClosure}


def build_model(config: ModelConfig, key: PRNGKeyArray) -> Emulator:
    if config.formulation == "aeff":
        return ExchangeRescale()
    energy = config.formulation == "energy"
    if energy and config.fields:
        raise ValueError("the energy closure reads m only: set fields: false")
    network = ARCHS[config.arch](
        num_spatial_dims=2,
        # m [, h_ex, h_d] + the embedded scalars (eps alone for the energy)
        in_channels=(9 if config.fields else 3) + (1 if energy else 3),
        out_channels=1 if energy else 3,
        hidden_channels=config.hidden_channels,
        num_blocks=config.num_blocks,
        activation=jax.nn.gelu,
        boundary_mode="neumann",
        use_norm=False,  # GroupNorm would make the network non-local
        key=key,
    )
    if config.zero_init:
        network = eqx.tree_at(
            lambda n: (n.projection.weight, n.projection.bias),
            network,
            replace_fn=jnp.zeros_like,
        )
    return FORMULATIONS[config.formulation](network, config.fields)


def load_config(run: Path) -> ModelConfig:
    if (run / "config.yaml").exists():
        return ModelConfig(**yaml.safe_load((run / "config.yaml").read_text())["model"])
    # runs from before the recipes were YAML files
    meta = json.loads((run / "metadata.json").read_text())
    if not meta.get("closure"):
        raise ValueError(f"{run} is not a closure model")
    fields = ("arch", "hidden_channels", "num_blocks", "zero_init")
    return ModelConfig(
        formulation="closure", **{name: meta[name] for name in fields if name in meta}
    )


def save_model(model: Emulator, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_model(run: Path, tag: str = "model") -> Emulator:
    skeleton = build_model(load_config(run), jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(run / f"{tag}.eqx", skeleton)
    return eqx.nn.inference_mode(model)
