from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llg_emulator.config import dataset_dir


@dataclass
class TrainConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    size: Literal["small", "med", "large"]
    checkpoint_every: int

    @property
    def sp4_path(self) -> Path:
        return dataset_dir(split="sp4", size=self.size) / "sample-00001-of-00001"

    @property
    def large_scale_path(self) -> Path:
        """Same physics as sp4 on an 8x wider mesh; the domain-size check."""
        return dataset_dir(split="large-scale", size=self.size) / "sample_0"
