"""Coarse micromagnetics baselines B1 and B3, relaxed with magnum.np's energy minimiser.

B1: coarse micromagnetics with the physical exchange A. On cells far larger than the exchange
    length a 180 degree wall across one cell costs only ~4A/dx per area, so walls are nearly free
    and domains over-fragment.
B3: Suess's coarse-grained domain solver (PDF section 10). It replaces exchange by a wall energy
    sigma per cell face, which is the same discrete exchange with A_eff = c sigma dx. The PDF is not
    consistent about c (its eq. 5 gives 1/4, its eq. 7 gives 1/2); c = 1/2 reproduces its <m_y>
    better (baselines/check_coarse_pdf.py), so that is WALL_FACTOR.

Both relax to the nearest energy minimum from a given start, so they keep domains and history.
"""

from collections.abc import Sequence

import numpy as np
import torch
from magnumnp import DemagField, ExchangeField, ExternalField, Mesh, MinimizerBB, State
from scipy import constants

from shield.geometry import MS

SIGMA = 4.6e-3  # J/m^2, the PDF's wall energy for NiFe (section 10.2)
WALL_FACTOR = 0.5  # B3: A_eff = WALL_FACTOR * SIGMA * dx
MAX_ITERATIONS = 100_000
TORQUE_TOLERANCE = 1.0  # A/m, max |m x (m x h)| at convergence (magnum.np's default)


def solve(
    fraction: np.ndarray, dx: float, b_ext: Sequence[float], m0: np.ndarray, A: float
) -> torch.Tensor:
    """Relaxed magnetisation M (nx, ny, nz, 3) in A/m, starting from the unit field m0 (nx, ny, nz, 3).

    fraction: magnet volume fraction (nx, ny, nz) on cubic cells of size dx (m).
    b_ext: uniform applied field mu0 H_ext (3,) in tesla.
    A: exchange constant (J/m): the physical A for B1, c * SIGMA * dx for B3.
    """
    state = State(Mesh(fraction.shape, (dx, dx, dx)))
    volume = torch.as_tensor(fraction, device=state.device)[..., None]  # (nx, ny, nz, 1)
    magnet = volume > 0
    # A = 0 in air decouples the magnet from air cells, and m = 0 there stays 0
    state.material = {"Ms": MS * volume, "A": A * magnet}
    state.m = magnet * torch.as_tensor(m0, device=state.device)
    h_ext = [b / constants.mu_0 for b in b_ext]

    minimizer = MinimizerBB([DemagField(), ExchangeField(), ExternalField(h_ext)])
    if not minimizer.minimize(state, maxiter=MAX_ITERATIONS, dm_tol=TORQUE_TOLERANCE):
        raise RuntimeError("energy minimisation did not converge")
    return MS * volume * state.m
