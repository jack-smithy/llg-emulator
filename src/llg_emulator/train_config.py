from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from llg_emulator.config import dataset_dir


@dataclass
class TrainConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    size: Literal["small", "med", "large"]
    checkpoint_every: int
    cpu_buffer_size: int
    device_buffer_size: int

    @property
    def sp4_path(self) -> Path:
        return dataset_dir(split="sp4", size=self.size) / "sample_0"
