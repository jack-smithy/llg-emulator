import jax


from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_blocks: int = 4
    num_modes: int = 32
    activation: Callable = jax.nn.gelu
