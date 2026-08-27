from dataclasses import dataclass
from pathlib import Path

from llg_emulator.config import DATA_ROOT, sp4_path


@dataclass
class TrainConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    checkpoint_every: int

    @property
    def sp4_path(self) -> Path:
        return sp4_path()

    @property
    def large_scale_path(self) -> Path:
        """Same physics as sp4 on a wider mesh; the domain-size check.
        Not present in the current dataset — sp4 itself now exercises a
        different mesh (100x25 cells vs the 256x256 training mesh)."""
        return DATA_ROOT / "val" / "large_scale"
