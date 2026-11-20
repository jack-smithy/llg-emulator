"""Checks for the Maxwell solver. Run inside a job, e.g.
srun -p devcpu -c 4 -t 15 uv run pytest -q tests/test_maxwell.py
"""

import numpy as np
import pytest
import torch
from scipy import constants, optimize

from baselines import maxwell
from shield import coarsen, geometry, strayfield
from shield.geometry import MS


def law(H: torch.Tensor, chi: float) -> torch.Tensor:
    """The PDF's eq. 1 written forwards: M (..., 3) from H (..., 3)."""
    norm = H.norm(dim=-1, keepdim=True)
    x = chi * norm / MS
    return MS * x / torch.sqrt(1 + x**2) * H / norm


@pytest.mark.parametrize("b", [0.01, 0.1, 2.0])
def test_single_cell_matches_scalar_equation(b):
    """A single cubic cell is uniformly magnetised, M = law(H_ext - M / 3) along the field."""
    chi = 1e3
    M = maxwell.solve(np.ones((1, 1, 1)), 10e-9, (b, 0, 0), chi)
    h_ext = b / constants.mu_0

    def residual(m):
        x = chi * (h_ext - m / 3) / MS
        return m - MS * x / np.sqrt(1 + x**2)

    expected = optimize.brentq(residual, 0, min(MS, 3 * h_ext), xtol=1e-12)
    assert float(M[0, 0, 0, 0]) == pytest.approx(expected, rel=1e-6)
    assert float(M[0, 0, 0, 1:].abs().max()) < 1e-9 * MS


def test_law_holds_in_every_cell():
    """Ellipse plate with partly filled edge cells: M = fraction * law(H_ext + H_demag(M))."""
    chi = 1e4
    dx = 10e-9
    fine = geometry.volume_fraction("ellipse", (120e-9, 60e-9, 20e-9), dx / 2)
    fraction = coarsen.block_average(torch.as_tensor(fine)[..., None], 2)[..., 0].cpu().numpy()
    assert ((fraction > 0) & (fraction < 1)).any()  # partly filled edge cells
    b_ext = (0.02, 0.05, 0.01)
    M = maxwell.solve(fraction, dx, b_ext, chi)

    h_ext = torch.as_tensor(b_ext) / constants.mu_0
    H = h_ext + strayfield.stray_field_operator(fraction.shape, dx, pad=(0, 0, 0))(M)
    expected = torch.as_tensor(fraction)[..., None] * law(H, chi)
    # the solver stops at |H_law - H| < 1e-8 |H_ext|, which moves M by up to ~chi times that
    torch.testing.assert_close(M, expected, rtol=1e-5, atol=1e-4 * MS)


def test_zero_field_gives_zero_magnetisation():
    M = maxwell.solve(np.ones((4, 2, 2)), 10e-9, (0, 0, 0))
    assert float(M.abs().max()) == 0
