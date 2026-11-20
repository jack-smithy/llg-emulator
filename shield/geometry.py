"""Shield material and shapes on a regular grid.

All shapes are plates extruded along z and centred on the origin, as in the PDF
(the 100 x 50 x 20 um shield spans x in [-50, 50] um, y in [-25, 25] um, z in [-10, 10] um).
"""

from collections.abc import Sequence

import numpy as np
from scipy import constants

# NiFe as in Shield_Optimization_DS.pdf: J_s = mu0 Ms = 1 T, A = 13 pJ/m, no anisotropy
MS = 1.0 / constants.mu_0  # A/m (~7.96e5)
A_EX = 13e-12  # J/m
EXCHANGE_LENGTH = np.sqrt(2 * A_EX / (constants.mu_0 * MS**2))  # ~5.7 nm


def n_cells(size: Sequence[float], dx: float) -> tuple[int, int, int]:
    """Cells per axis for a box of `size` (m) with cell size `dx` (m); the box must fit exactly."""
    n = np.rint(np.array(size) / dx).astype(int)
    if not np.allclose(n * dx, size, rtol=1e-9, atol=0):
        raise ValueError(f"box {size} is not a whole number of {dx} m cells")
    return tuple(int(k) for k in n)


def volume_fraction(kind: str, size: Sequence[float], dx: float) -> np.ndarray:
    """Magnet mask (nx, ny, nz), 1 inside and 0 outside, of a plate of `size` (m) on cells of `dx` (m).

    kind: "prism" (the full box) or "ellipse" (an ellipse touching the box edges, a disk when
    Lx == Ly). A cell counts as magnet when its centre is inside. Partly filled cells at curved
    edges appear when the fine mask is block-averaged to coarser cells.
    """
    nx, ny, nz = n_cells(size, dx)
    if kind == "prism":
        area = np.ones((nx, ny))
    elif kind == "ellipse":
        x, y = np.meshgrid(cell_centres(nx, dx), cell_centres(ny, dx), indexing="ij")
        area = (2 * x / size[0]) ** 2 + (2 * y / size[1]) ** 2 <= 1
    else:
        raise ValueError(f"unknown shape {kind!r}")
    return np.repeat(area[:, :, None], nz, axis=2).astype(float)


def cell_centres(n: int, dx: float) -> np.ndarray:
    """Cell-centre coordinates (m) along one axis of n cells, centred on 0."""
    return (np.arange(n) + 0.5) * dx - n * dx / 2
