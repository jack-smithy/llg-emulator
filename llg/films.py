from dataclasses import dataclass
from pathlib import Path

from llg.constants import DATASETS


@dataclass(frozen=True)
class Film:
    cells: int  # 5 nm cells per side; every film is square
    split: str
    pattern: str  # file glob in data/<split>
    factors: tuple[int, ...]  # coarse-graining factors it is evaluated at
    sp4: bool = False  # one trajectory from SP4's s-state under SP4 field 1

    @property
    def side_um(self) -> float:
        return self.cells * 5e-3

    def files(self, dataset: str) -> list[Path]:
        return sorted((DATASETS / dataset / "data" / self.split).glob(self.pattern))


FILMS = {
    "sq64": Film(64, "test", "llg_test_sq64.hdf5", (1, 2, 4)),
    "sq128": Film(128, "test", "llg_test_sq128.hdf5", (1, 2, 4, 8)),
    "sq256": Film(256, "test", "llg_test_[0-9]*.hdf5", (1, 2, 4, 8, 16)),
    "sq512": Film(512, "test", "llg_test_sq512.hdf5", (2, 4, 8, 16, 32)),
    "sq1024": Film(1024, "test", "llg_test_sq1024.hdf5", (2, 4, 8, 16, 32)),
    "large": Film(6144, "test", "llg_test_large6144.hdf5", (4, 8, 16, 32, 64), sp4=True),
    # the validation shards: rollout scores for model selection
    "valid256": Film(256, "valid", "llg_valid_[0-9]*.hdf5", (1, 2, 4, 8, 16)),
}
