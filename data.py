import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import h5py
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import torch
from jaxtyping import Array, Float
from the_well.data import WellDataset
from torch.utils.data import DataLoader, default_collate

N_STEPS = 100  # frames after the initial one in every trajectory (1 ns)
FIELDS = ("mx", "my", "mz")
SCALARS = ("Ms", "A", "alpha", "Hx", "Hy", "Hz")  # order of `constant_scalars`

DATASETS = Path("datasets")
RESULTS = Path(os.environ.get("RESULTS_ROOT", "results-v2"))


def run_dir(dataset: str, run: str) -> Path:
    return RESULTS / dataset / run


def cache_dir(dataset: str) -> Path:
    return RESULTS / dataset / "scaling"


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


class Batch(NamedTuple):
    m: Float[Array, "..."]  # (B, nx, ny, 3)
    targets: Float[Array, "..."]  # (B, T, nx, ny, 3), the T frames after m
    h: Float[Array, "..."]  # (B, 3) applied field in A/m


def well_dataset(dataset: str, split: str, n_steps: int) -> WellDataset:
    data = WellDataset(
        path=str(DATASETS / dataset),
        well_split_name=split,
        n_steps_input=1,
        n_steps_output=n_steps,
        use_normalization=False,
    )
    assert tuple(data.metadata.constant_scalar_names) == SCALARS
    return data


def well_loader(
    dataset: str, split: str, n_steps: int, batch_size: int, seed: int = 0, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        well_dataset(dataset, split, n_steps),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=4,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=numpy_collate,
    )


def numpy_collate(samples: list[dict]) -> dict:
    return jtu.tree_map(np.asarray, default_collate(samples))


def to_batch(sample: dict) -> Batch:
    return Batch(
        m=sample["input_fields"][:, 0],
        targets=sample["output_fields"],
        h=sample["constant_scalars"][:, 3:6],
    )


def coarse_grain(m: Float[Array, "..."], k: int) -> Float[Array, "..."]:
    """Block mean onto k-times larger cells, back on the unit sphere; numpy or jax."""
    if k == 1:
        return m
    *lead, nx, ny, _ = m.shape
    m = m.reshape(*lead, nx // k, k, ny // k, k, 3).mean(axis=(-4, -2))
    return m / ((m**2).sum(axis=-1, keepdims=True) ** 0.5)


def d4(m: Float[Array, "..."], h: Float[Array, "..."], g: int):
    """Element g of the square's symmetry group: g % 4 quarter turns, then for g >= 4 a
    mirror x -> -x. m and h are axial vectors, so the mirror flips their y and z."""
    m = np.rot90(m, g % 4, axes=(-3, -2))
    c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[g % 4]
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], m.dtype)
    m, h = m @ rotation.T, h @ rotation.T
    if g >= 4:
        mirror = np.array([1, -1, -1], m.dtype)
        m, h = m[..., ::-1, :, :] * mirror, h * mirror
    return m, h


def prepare(batch: Batch, k: int, g: int | None = None) -> Batch:
    """On the device, augmented by d4 element g, then coarse-grained by k."""
    m, targets, h = batch
    if g is not None:
        m, h_aug = d4(m, h, g)
        targets, _ = d4(targets, h, g)
        h = h_aug
    return Batch(
        m=coarse_grain(jnp.asarray(m), k),
        targets=coarse_grain(jnp.asarray(targets), k),
        h=jnp.asarray(h),
    )


def read_trajectories(
    paths: list[Path],
) -> Iterator[tuple[Callable[[int], np.ndarray], np.ndarray]]:
    """(frame reader, H) per trajectory; frames are read one at a time from disk."""
    for path in paths:
        with h5py.File(path) as f:
            for j in range(f["t0_fields"]["mx"].shape[0]):
                h = np.array([f["scalars"][name][j] for name in SCALARS[3:]])

                def read(t, f=f, j=j):
                    return np.stack([f["t0_fields"][c][j, t] for c in FIELDS], axis=-1)

                yield read, h
