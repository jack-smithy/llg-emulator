"""Training core: model construction, the loss, the jitted update step, and
checkpointing.

The model is pdequinox's `ClassicResNet` wrapped in
`ConstantEmbeddingMetadataNetwork`, which appends the conditioning vector
(`[Hx, Hy] / Ms`, see `utils.conditioning`) to the input as constant channels.
The objective is a plain one-step MSE on `(m_t, m_{t+1})` pairs.
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from pdequinox.arch import Flower
from model import ResidualEmulator
from model_config import ModelConfig

COND_DIM = 2  # [Hx, Hy] / Ms


def build_model(config: ModelConfig, key) -> eqx.Module:
    network = Flower(
        num_spatial_dims=2,
        in_channels=3 + COND_DIM,  # m (3) + the embedded conditioning
        out_channels=3,
        hidden_channels=config.hidden_channels,
        num_blocks=config.num_blocks,
        num_levels=config.num_levels,
        num_heads=config.num_heads,
        activation=config.activation,
        boundary_mode="neumann",  # finite thin film, not periodic
        key=key,
    )
    model = ResidualEmulator(network=network)

    return model


@eqx.filter_jit
def loss_fn(model, m0, m1, cond):
    return jnp.mean(jnp.square(jax.vmap(model)(m0, cond) - m1))


@eqx.filter_jit(donate="all-except-first")
def update_fn(model: eqx.Module, batch: PyTree, optimizer, opt_state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, *batch)
    updates, opt_state = optimizer.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def save_model(model: eqx.Module, config: ModelConfig, path: Path, tag: str):
    """Serialise the weights to `<path>/<tag>.eqx` + `metadata.json`."""
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / f"{tag}.eqx", model)
    meta = {
        "hidden_channels": config.hidden_channels,
        "num_blocks": config.num_blocks,
        "num_levels": config.num_levels,
        "num_heads": config.num_heads,
    }
    # rewrite every time: a stale sidecar builds the wrong skeleton on load
    (path / "metadata.json").write_text(json.dumps(meta))


def load_model(path: Path, key, tag: str) -> eqx.Module:
    """Rebuild the model from `<tag>.eqx` + `metadata.json`. `key` only seeds
    the skeleton whose weights are then overwritten."""
    meta = json.loads((path / "metadata.json").read_text())
    config = ModelConfig(
        hidden_channels=meta["hidden_channels"],
        num_blocks=meta["num_blocks"],
        num_heads=meta["num_heads"],
        num_levels=meta["num_levels"],
    )
    return eqx.tree_deserialise_leaves(path / f"{tag}.eqx", build_model(config, key))
