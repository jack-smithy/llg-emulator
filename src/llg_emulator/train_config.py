from dataclasses import dataclass
from typing import Literal


@dataclass
class TrainConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    size: Literal["small", "med", "large"]
    checkpoint_every: int
    cpu_buffer_size: int
    device_buffer_size: int
