import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy.linalg as LA
import pdequinox as pdeqx
from pdequinox.arch import ClassicFNO

COND_DIM = 2  # [Hx, Hy] / Ms


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_blocks: int = 4
    num_modes: int = 32
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


def build_model(config: ModelConfig, key) -> eqx.Module:
    network = ClassicFNO(
        num_spatial_dims=2,
        in_channels=5 + COND_DIM,  # m (3) + coords (2) + the embedded conditioning
        out_channels=3,
        hidden_channels=config.hidden_channels,
        num_blocks=config.num_blocks,
        activation=config.activation,
        num_modes=config.num_modes,
        boundary_mode="neumann",  # finite thin film, not periodic
        key=key,
    )
    model = ResidualEmulator(network=network)

    return model


def save_model(model: eqx.Module, config: ModelConfig, path: Path, tag: str):
    """Serialise the weights to `<path>/<tag>.eqx` + `metadata.json`."""
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / f"{tag}.eqx", model)
    meta = {
        "hidden_channels": config.hidden_channels,
        "num_blocks": config.num_blocks,
        "num_modes": config.num_modes,
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
        num_modes=meta["num_modes"],
    )
    return eqx.tree_deserialise_leaves(path / f"{tag}.eqx", build_model(config, key))
