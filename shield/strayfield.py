"""Stray field of a magnetisation on a regular grid, using magnum.np's FFT demag field.

magnum.np's DemagField (Newell's tensor, FFT convolution, open boundaries) is exact for any
cell-wise constant magnetisation at any grid size. Padding the magnet with air cells (Ms = 0)
gives the field outside it, at the probe points and on planes above the shield.

Importing magnum.np switches torch to float64 and makes the GPU (if any) the default device.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch
from magnumnp import DemagField, Mesh, State
from scipy.interpolate import RegularGridInterpolator

from shield.geometry import cell_centres

PROBE_Y = 0.44  # probe y as a fraction of Ly: 22 um on the PDF's 50 um wide shield
PROBE_HEIGHTS = (0.1, 0.25, 0.5, 1.0)  # above the top / below the bottom surface, fraction of Lz


def probe_points(size: Sequence[float]) -> np.ndarray:
    """Probe positions (16, 3) in m for a shield of size (Lx, Ly, Lz) centred on the origin.

    x = 0, y = +-0.44 Ly, z = +-(Lz/2 + h Lz). On the PDF's 100 x 50 x 20 um shield, h = 0.1 and
    h = 0.5 are its probes (0, +-22, +-12) um and (0, +-22, +-20) um.

    Sampled from cell averages, the nearest probe sits 0.1 nz cells from the surface; at two cells
    it is off by ~1.5 %, and the error falls roughly with the square of the distance in cells. For
    thin magnets, upsample M by an integer factor first (exact for the FFT method) so that this
    probe is several cells away.
    """
    _, ly, lz = size
    points = []
    for z_sign in (1, -1):
        for h in PROBE_HEIGHTS:
            for y_sign in (1, -1):
                points.append((0.0, y_sign * PROBE_Y * ly, z_sign * (lz / 2 + h * lz)))
    return np.array(points)


def probe_padding(n: Sequence[int]) -> tuple[int, int, int]:
    """Air cells (px, py, pz) per side that put every probe between cell centres of the padded grid.

    The probes lie above the footprint, but y = 0.44 Ly can be beyond the last cell centre of a
    narrow grid, so y gets one air cell; z gets room for the highest probe.
    """
    nz = n[2]
    return (0, 1, int(np.ceil(max(PROBE_HEIGHTS) * nz)) + 2)


def stray_field_operator(
    n: Sequence[int], dx: float, pad: Sequence[int]
) -> Callable[[np.ndarray | torch.Tensor], torch.Tensor]:
    """Return a function M -> H for a magnet of n = (nx, ny, nz) cubic cells of size dx (m).

    The magnet sits in the middle of a grid padded with pad = (px, py, pz) air cells per side; the
    padded grid is centred on the origin like the magnet. The function takes M (nx, ny, nz, 3) in
    A/m and returns H (nx + 2 px, ny + 2 py, nz + 2 pz, 3) in A/m. The demag kernel is computed on
    the first call and reused, so make one operator per grid and call it for every frame.
    """
    padded = [k + 2 * p for k, p in zip(n, pad)]
    origin = [-(k / 2 + p) * dx for k, p in zip(n, pad)]
    state = State(Mesh(padded, (dx, dx, dx), origin))
    state.material = {"Ms": 1.0}  # so that state.m is M itself
    demag = DemagField()
    inside = tuple(slice(p, p + k) for k, p in zip(n, pad))

    def field(M: np.ndarray | torch.Tensor) -> torch.Tensor:
        M_padded = torch.zeros(padded + [3], dtype=state.dtype, device=state.device)
        M_padded[inside] = torch.as_tensor(M, dtype=state.dtype, device=state.device)
        state.m = M_padded
        return demag.h(state)

    return field


def sample(H: torch.Tensor, dx: float, points: np.ndarray) -> np.ndarray:
    """Trilinear interpolation of H (nx, ny, nz, 3) at points (k, 3) in m; returns (k, 3).

    H lives on cubic cells of size dx centred on the origin (as returned by stray_field_operator).
    A point at a cell centre gets exactly that cell's value; points outside the grid raise.
    """
    axes = [cell_centres(n, dx) for n in H.shape[:3]]
    return RegularGridInterpolator(axes, H.cpu().numpy())(points)
