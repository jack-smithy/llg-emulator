import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as LA
import pdequinox as pdeqx
from pdequinox.arch import ClassicFNO, DilatedResNet


@dataclass
class ModelConfig:
    arch: str = "fno"  # "fno" | "dilated"
    hidden_channels: int = 32
    num_blocks: int = 4
    num_modes: int = 32  # fno only
    use_demag: bool = False  # feed the exact demag field as 3 input channels
    # 2 geometry channels (utils.spatial_features): "absolute" cell-centre
    # coordinates in um, bounded "edge" distances, or "none"
    coords: str = "absolute"
    # condition on log(delta / l_ex) as well as the field (utils.conditioning)
    cell_size_cond: bool = False
    # a learned correction to micromagnetics on the model's own mesh: the input
    # starts with the physics.CoarseLLG prediction, which the output is added to
    use_solver: bool = False
    # GroupNorm in the dilated ResNet's blocks (pdequinox's default): it normalises
    # each feature over the whole film, which makes the network non-local — its
    # output at a cell depends on the rest of the film. Off = strictly local.
    use_norm: bool = True
    # the network predicts the per-step increment in units of this (the training
    # data's std of one-step differences), so its output is O(1) from the start
    output_scale: float = 1.0
    activation: Callable = jax.nn.gelu

    @property
    def cond_dim(self):
        return 2 + self.cell_size_cond  # [Hx, Hy] / Ms [+ log(delta / l_ex)]


class ResidualEmulator(eqx.Module):
    network: eqx.Module
    output_scale: float = eqx.field(static=True)

    def __init__(self, network, output_scale: float = 1.0):
        # utils.conditioning already scales the conditioning to O(1)
        self.network = pdeqx.ConstantEmbeddingMetadataNetwork(
            network=network, normalization_factor=1.0
        )
        self.output_scale = output_scale

    def __call__(self, m0, meta_data):
        dm = self.output_scale * self.network(m0, meta_data=meta_data)  # type: ignore
        m1 = m0[:3] + dm
        return m1 / LA.norm(m1, axis=0, keepdims=True)


def build_model(config: ModelConfig, key) -> eqx.Module:
    # m (3) [+ demag (3)] [+ coords (2)] + the embedded conditioning
    in_channels = (
        3
        + (3 if config.use_solver else 0)
        + (3 if config.use_demag else 0)
        + (2 if config.coords != "none" else 0)
        + config.cond_dim
    )
    if config.arch == "fno":
        # NB the spectral weights are indexed by integer mode number on the
        # domain, so with a fixed cell size they are only physically scaled
        # right on films the size of the training set; boundary_mode is
        # accepted but ignored (ClassicFNO is periodic).
        network = ClassicFNO(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=3,
            hidden_channels=config.hidden_channels,
            num_blocks=config.num_blocks,
            activation=config.activation,
            num_modes=config.num_modes,
            boundary_mode="neumann",
            key=key,
        )
    elif config.arch == "dilated":
        # fixed receptive field in cells = fixed physical receptive field
        # (every film shares the cell size), so the weights transfer to any
        # geometry; the long-range demag physics comes in through use_demag
        network = DilatedResNet(
            num_spatial_dims=2,
            in_channels=in_channels,
            out_channels=3,
            hidden_channels=config.hidden_channels,
            num_blocks=config.num_blocks,
            activation=config.activation,
            boundary_mode="neumann",  # reflect padding: no flux through edges
            use_norm=config.use_norm,
            key=key,
        )
    else:
        raise ValueError(f"unknown arch {config.arch!r}")
    model = ResidualEmulator(network=network, output_scale=config.output_scale)
    if config.use_solver:
        # zero correction at initialisation: the model starts as the solver
        # itself, and training can only move it away where that helps
        proj = model.network.network.projection
        model = eqx.tree_at(
            lambda m: (
                m.network.network.projection.weight,
                m.network.network.projection.bias,
            ),
            model,
            (jnp.zeros_like(proj.weight), jnp.zeros_like(proj.bias)),
        )
    return model


def save_model(model: eqx.Module, config: ModelConfig, path: Path, tag: str):
    """Serialise the weights to `<path>/<tag>.eqx` + `metadata.json`."""
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / f"{tag}.eqx", model)
    meta = {
        "arch": config.arch,
        "hidden_channels": config.hidden_channels,
        "num_blocks": config.num_blocks,
        "num_modes": config.num_modes,
        "use_demag": config.use_demag,
        "coords": config.coords,
        "cell_size_cond": config.cell_size_cond,
        "use_solver": config.use_solver,
        "use_norm": config.use_norm,
        "output_scale": config.output_scale,
    }
    # rewrite every time: a stale sidecar builds the wrong skeleton on load
    (path / "metadata.json").write_text(json.dumps(meta))


def load_model(path: Path, key, tag: str) -> tuple[eqx.Module, ModelConfig]:
    """Rebuild the model from `<tag>.eqx` + `metadata.json`. `key` only seeds
    the skeleton whose weights are then overwritten. Also returns the config,
    which the eval scripts need to build the matching inputs. Sidecars from
    before `coords` / `cell_size_cond` / `output_scale` describe models whose
    conditioning was scaled differently, so they are refused rather than
    rebuilt wrongly."""
    meta = json.loads((path / "metadata.json").read_text())
    if "coords" not in meta:
        raise ValueError(f"{path}: checkpoint predates the current input layout")
    config = ModelConfig(
        arch=meta["arch"],
        hidden_channels=meta["hidden_channels"],
        num_blocks=meta["num_blocks"],
        num_modes=meta["num_modes"],
        use_demag=meta["use_demag"],
        coords=meta["coords"],
        cell_size_cond=meta["cell_size_cond"],
        use_solver=meta.get("use_solver", False),
        use_norm=meta.get("use_norm", True),
        output_scale=meta["output_scale"],
    )
    model = eqx.tree_deserialise_leaves(path / f"{tag}.eqx", build_model(config, key))
    return model, config
