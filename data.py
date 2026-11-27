"""The stored truth on the coarse grid. One HDF5 file per loop, datasets/<geometry>/<split>/loop_NN.h5
(datagen.save_loop): m (T, X, Y, Z, 3), block averages on the stored grid; B (T, 3) [T]; attributes
dx, fine_dx, Ms, A, axis. On the coarse grid every cell is block-averaged by a factor and renormalised
to |m| = 1."""

from pathlib import Path

import h5py
import numpy as np
import torch
from magnumnp.common.constants import mu_0
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from physics import DEVICE

DATASETS = Path("datasets")
GEOMETRIES = ["G1", "G2", "G3"]  # the training geometries, folders under DATASETS


def loop_files(geometry: str, split: str) -> list[Path]:
    return sorted(Path(DATASETS, geometry, split).glob("*.h5"))


def coarse_grid(geometry: str, factor: int) -> tuple[list[int], list[float]]:
    """Cells and cell size [m] of a geometry after block-averaging the stored data by factor."""
    with h5py.File(loop_files(geometry, "train")[0]) as f:
        n, dx = list(f["m"].shape[1:4]), float(f.attrs["dx"])
    assert all(k % factor == 0 for k in n), (
        f"{factor} does not divide the stored grid {n} of {geometry}"
    )
    return [k // factor for k in n], [dx * factor] * 3


def coarsen(m: Tensor, factor: int) -> Tensor:
    """Block-average m (B, X, Y, Z, 3) over factor^3 cells and renormalise to |m| = 1."""
    m = F.avg_pool3d(m.permute(0, 4, 1, 2, 3), factor).permute(0, 2, 3, 4, 1)
    return F.normalize(m, dim=-1)


class StoredStates(Dataset):
    """Every stored state of every loop of the geometries, on the coarse grid, padded with m = 0 at the
    high-index ends to the largest grid among them, so that batches can mix geometries. A sample:
    {"m": (X, Y, Z, 3), "mask": (X, Y, Z), 1 in the magnet and 0 in the padding, "h_ext": (3,) [A/m]},
    the relaxed fine state at its applied field."""

    def __init__(self, geometries: list[str], split: str, factor: int):
        self.factor = factor
        self.states = []  # (file, step)
        for geometry in geometries:
            for path in loop_files(geometry, split):
                with h5py.File(path) as f:
                    self.states += [(path, t) for t in range(f["m"].shape[0])]

        shape, dx = zip(*[coarse_grid(geometry, factor) for geometry in geometries])
        self.shape = np.max(shape, axis=0).tolist()
        assert all(d == dx[0] for d in dx), f"the geometries have different cell sizes: {dx}"
        self.dx = dx[0]  # coarse cell size [m] along x, y, z: the stored cell size times factor

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        path, t = self.states[index]
        with h5py.File(path) as f:
            m = torch.as_tensor(f["m"][t])
            h_ext = torch.as_tensor(f["B"][t] / mu_0, dtype=torch.float32)
        m = coarsen(m[None], self.factor)[0]
        padding = [p - k for p, k in zip(self.shape, m.shape[:3])]
        mask = F.pad(
            torch.ones(m.shape[:3]), (0, padding[2], 0, padding[1], 0, padding[0])
        )
        m = F.pad(m, (0, 0, 0, padding[2], 0, padding[1], 0, padding[0]))
        return {"m": m, "mask": mask, "h_ext": h_ext}


def load_loops(geometry: str, split: str, factor: int) -> list[dict]:
    """Every loop of a split, unpadded: {"name", "m": coarse (T, X, Y, Z, 3), "fields": (T, 3) [A/m],
    "axis": (3,)}. Loops from a random start have fewer steps than loops from saturation."""
    loops = []
    for path in loop_files(geometry, split):
        with h5py.File(path) as f:
            m = torch.as_tensor(f["m"][:], device=DEVICE)
            fields = torch.as_tensor(
                f["B"][:] / mu_0, device=DEVICE, dtype=torch.float32
            )
            axis = np.array(f.attrs["axis"])
        loops.append(
            {
                "name": f"{geometry}/{split}/{path.stem}",
                "m": coarsen(m, factor),
                "fields": fields,
                "axis": axis,
            }
        )
    return loops
