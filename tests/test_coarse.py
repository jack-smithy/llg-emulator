"""Checks for the coarse micromagnetics baselines. Run inside a job, e.g.
srun -p devcpu -c 4 -t 15 uv run pytest -q tests/test_coarse.py
"""

import numpy as np
import torch

from baselines import coarse
from shield import geometry, protocols
from shield.geometry import MS


def test_single_cell_turns_along_the_field():
    """A cube has no shape anisotropy, so its one cell turns fully along any applied field."""
    b_ext = np.array([0.01, 0.02, -0.005])
    m0 = np.array([[[[0.0, 0.0, 1.0]]]])
    M = coarse.solve(np.ones((1, 1, 1)), 10e-9, b_ext, m0, A=geometry.A_EX)
    expected = MS * b_ext / np.linalg.norm(b_ext)
    np.testing.assert_allclose(M[0, 0, 0].cpu().numpy(), expected, atol=1e-4 * MS)


def test_strong_field_saturates_and_air_stays_empty():
    fraction = geometry.volume_fraction("ellipse", (160e-9, 80e-9, 20e-9), 10e-9)
    m0 = protocols.initial_state("random", fraction.shape, np.random.default_rng(0))
    M = coarse.solve(fraction, 10e-9, (0, 2.0, 0), m0, A=geometry.A_EX)

    inside = torch.as_tensor(fraction > 0, device=M.device)
    assert float(M[~inside].abs().max()) == 0
    assert float(M[inside][:, 1].min()) > 0.999 * MS
