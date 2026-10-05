import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as LA
import pdequinox as pdeqx
from pdequinox.arch import DilatedResNet

COND_DIM = 3  # [Hx, Hy] / Ms, log(delta / l_ex)


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_blocks: int = 2
    # solver-in-the-loop (Um et al. 2020): the network corrects the coarse solver's
    # step, m' = P(m) + C(P(m)), and sees only P(m); otherwise it maps m + coords
    solver_in_the_loop: bool = False
    # the latest frame's exact demag field (physics.demag_cache) as 3 more inputs
    use_demag: bool = False
    # frames of context: the latest m plus the in_frames - 1 before it
    in_frames: int = 1
    # zero the output projection at init, so the untrained model adds exactly nothing
    # (for solver-in-the-loop: starts as the coarse solver itself)
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


def in_channels(config: ModelConfig) -> int:
    """P(m) (3) for solver-in-the-loop; otherwise the in_frames frames (3 each) +
    coords (2) [+ demag (3)]; plus the embedded conditioning."""
    if config.solver_in_the_loop:
        if config.use_demag or config.in_frames != 1:
            raise ValueError("solver-in-the-loop sees P(m) alone: no demag, one frame")
        return 3 + COND_DIM
    return 3 * config.in_frames + 2 + 3 * config.use_demag + COND_DIM


def build_model(config: ModelConfig, key) -> eqx.Module:
    # fixed receptive field in cells, so the weights transfer to any film size;
    # no GroupNorm, which normalises over the whole film and makes the net non-local
    network = DilatedResNet(
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
    return ResidualEmulator(network=network)


def save_model(model: eqx.Module, config: ModelConfig, path: Path, tag: str):
    """Serialise the weights to `<path>/<tag>.eqx` + `metadata.json`."""
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / f"{tag}.eqx", model)
    meta = {
        "hidden_channels": config.hidden_channels,
        "num_blocks": config.num_blocks,
        "solver_in_the_loop": config.solver_in_the_loop,
        "use_demag": config.use_demag,
        "in_frames": config.in_frames,
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
        solver_in_the_loop=meta.get("solver_in_the_loop", False),
        use_demag=meta.get("use_demag", False),
        in_frames=meta.get("in_frames", 1),
        zero_init=meta.get("zero_init", False),
    )


def load_model(path: Path, key, tag: str) -> eqx.Module:
    """Rebuild the model from `<tag>.eqx` + `metadata.json`. `key` only seeds
    the skeleton whose weights are then overwritten."""
    model = build_model(load_config(path), key)
    return eqx.tree_deserialise_leaves(path / f"{tag}.eqx", model)
