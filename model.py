"""The learned-closure LLG: coarse micromagnetics stepped under one extra field,

    m' = LLG_dt[m; H_ext + H_theta(m)]
    H_theta(m) = eps Ms N(m, h_ex / Ms, h_d / Ms; H_ext / Ms, eps),   eps = l_ex / delta

The network only sees fields in units of Ms on the coarse mesh, plus eps, so coarser
meshes move its inputs towards zero rather than off the end of the training range, and
the eps prefactor sends the model back to plain coarse micromagnetics as cells grow.
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


class ClosureEmulator(eqx.Module):
    network: pdeqx.ConstantEmbeddingMetadataNetwork

    def __init__(self, network: eqx.Module):
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(network, 1.0)

    def closure_field(
        self, m: Float[Array, "..."], h: Float[Array, "..."], solver: LLGStepper
    ) -> Float[Array, "..."]:
        Ms = solver.Ms
        eps = jnp.sqrt(2 * solver.A / MU_0) / Ms / solver.dx[0]
        h_ex = exchange_field(m, solver.dx, Ms, solver.A)
        fields = jnp.concatenate([m, h_ex / Ms, solver.demag(m) / Ms], axis=-1)
        scalars = jnp.stack([h[0] / Ms, h[1] / Ms, eps])
        out = self.network(jnp.moveaxis(fields, -1, 0), meta_data=scalars)
        return eps * Ms * jnp.moveaxis(out, 0, -1)

    def __call__(
        self, m: Float[Array, "..."], h: Float[Array, "..."], solver: LLGStepper
    ) -> Float[Array, "..."]:
        return solver(m, h + self.closure_field(m, h, solver))


def build_model(config: ModelConfig, key: PRNGKeyArray) -> ClosureEmulator:
    network = ARCHS[config.arch](
        num_spatial_dims=2,
        in_channels=9 + 3,  # m, h_ex, h_d + the embedded scalars
        out_channels=3,
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
    return ClosureEmulator(network)


def load_config(run: Path) -> ModelConfig:
    if (run / "config.yaml").exists():
        return ModelConfig(**yaml.safe_load((run / "config.yaml").read_text())["model"])
    # runs from before the recipes were YAML files
    meta = json.loads((run / "metadata.json").read_text())
    if not meta.get("closure"):
        raise ValueError(f"{run} is not a closure model")
    fields = ("arch", "hidden_channels", "num_blocks", "zero_init")
    return ModelConfig(**{name: meta[name] for name in fields if name in meta})


def save_model(model: ClosureEmulator, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)


def load_model(run: Path, tag: str = "model") -> ClosureEmulator:
    skeleton = build_model(load_config(run), jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(run / f"{tag}.eqx", skeleton)
    return eqx.nn.inference_mode(model)
