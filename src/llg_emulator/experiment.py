"""Typed training configuration.

A run is fully specified by a TOML file. The resolved config (including
defaults) is logged to wandb as the run config (``asdict(cfg)``), so any run
can be reproduced and any checkpoint reloaded with the correct architecture via
``TrainConfig.from_dict(run.config)``.
"""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from typing import Callable, Literal

import jax
import optax


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_blocks: int = 4
    activation: str = "gelu"  # gelu | relu | tanh
    # demag-field input (physics.DemagField): mesh geometry the demag tensor
    # is precomputed for. Defaults match the dataset (256x256x1 film).
    mesh_n: tuple = (256, 256, 1)
    mesh_dx: tuple = (5e-9, 5e-9, 3e-9)
    demag_p: int = 20


@dataclass
class DataConfig:
    size: str = "small"  # small | med
    batch_size: int = 128
    viz_sample: str = "sp4/sample_0"  # rollout-viz trajectory


@dataclass
class OptimConfig:
    name: str = "adam"
    lr: float = 1e-3


@dataclass
class WandbConfig:
    project: str = "llg-emulator"
    entity: str | None = None
    mode: Literal["online", "offline", "disabled", "shared"] = (
        "online"  # online | offline | disabled
    )
    name: str | None = None  # run name; defaults to wandb's auto-generated name


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 10
    checkpoint_every: int = 8
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        d = dict(d)
        model = _build_section(ModelConfig, d.pop("model", {}), "model")
        data = _build_section(DataConfig, d.pop("data", {}), "data")
        optim = _build_section(OptimConfig, d.pop("optim", {}), "optim")
        wandb = _build_section(WandbConfig, d.pop("wandb", {}), "wandb")
        _check_keys(cls, d, "top-level")
        return cls(model=model, data=data, optim=optim, wandb=wandb, **d)

    @classmethod
    def from_toml(cls, path) -> "TrainConfig":
        with open(path, "rb") as f:
            return cls.from_dict(tomllib.load(f))


def _field_names(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


def _check_keys(cls, d: dict, section: str) -> None:
    unknown = set(d) - _field_names(cls)
    if unknown:
        raise ValueError(
            f"unknown key(s) in [{section}] config: {sorted(unknown)}; "
            f"allowed: {sorted(_field_names(cls))}"
        )


def _build_section(cls, d: dict, section: str):
    _check_keys(cls, d, section)
    return cls(**d)


_ACTIVATIONS = {"gelu": jax.nn.gelu, "relu": jax.nn.relu, "tanh": jax.nn.tanh}
_OPTIMIZERS = {"adam": optax.adam}


def build_activation(name: str) -> Callable:
    try:
        return _ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}"
        )


def build_optimizer(cfg: OptimConfig):
    try:
        factory = _OPTIMIZERS[cfg.name]
    except KeyError:
        raise ValueError(
            f"unknown optimizer {cfg.name!r}; choose from {sorted(_OPTIMIZERS)}"
        )
    return factory(cfg.lr)
