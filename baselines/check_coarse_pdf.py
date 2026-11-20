"""Reproduce the PDF's coarse-solver numbers (section 10.5, Table 3) for the 100 x 50 x 20 um shield.

PDF, 2 um cells, sigma = 4.6 mJ/m^2, B = 50 mT along y:
  - vortex start: <m_y> = 0.232, |mu0 H_z| = 70.7 mT at (0, 22, +-12) um, 33.0 / 33.1 mT at (0, 22, +-20) um;
  - uniform start (m along +y): <m_y> = 0.244.
Its eq. 5 and eq. 7 disagree by a factor 2 on the exchange this wall energy amounts to, so both
are run, and B1 (the physical A) on the same cells.

Run as a job, e.g.
  sbatch -t 0:30:00 --gres=gpu:a100:1 --wrap "uv run python -u -m baselines.check_coarse_pdf"
"""

import logging
import time

import numpy as np
from scipy import constants

from baselines import coarse
from shield import geometry, protocols, strayfield

SIZE = (100e-6, 50e-6, 20e-6)
DX = 2e-6
B = 0.05  # T, along y
# the stray field of M repeated onto finer cells is the same field (exact for the FFT method),
# and puts the probe 2 um above the surface 4 cells away instead of 1
UPSAMPLE = 4
CASES = {
    "B3, c = 1/4": 0.25 * coarse.SIGMA * DX,
    "B3, c = 1/2": 0.5 * coarse.SIGMA * DX,
    "B1, physical A": geometry.A_EX,
}
STARTS = ("vortex", "+y")
# rows of strayfield.probe_points: (0, 22, 12), (0, 22, -12), (0, 22, 20), (0, 22, -20) um
PROBES = (0, 8, 4, 12)


def probe_hz(M) -> np.ndarray:
    """mu0 H_z (mT) at the PROBES for the magnetisation M (nx, ny, nz, 3) on DX cells."""
    fine = M.repeat_interleave(UPSAMPLE, 0).repeat_interleave(UPSAMPLE, 1).repeat_interleave(UPSAMPLE, 2)
    dx = DX / UPSAMPLE
    n = fine.shape[:3]
    H = strayfield.stray_field_operator(n, dx, strayfield.probe_padding(n))(fine)
    H_probes = strayfield.sample(H, dx, strayfield.probe_points(SIZE))
    return 1e3 * constants.mu_0 * H_probes[PROBES, 2]


def main() -> None:
    logging.getLogger("magnum.np").setLevel(logging.WARNING)  # MinimizerBB logs every iteration
    fraction = geometry.volume_fraction("prism", SIZE, DX)
    rng = np.random.default_rng(0)
    print("PDF B3 (vortex): <m_y> 0.232, Hz 70.7 / -70.7 / 33.0 / -33.1 mT; B3 (uniform): <m_y> 0.244")
    print("case            start     <m_y>  Hz(22,12) Hz(22,-12) Hz(22,20) Hz(22,-20)  time (s)")
    for name, A in CASES.items():
        for start in STARTS:
            begin = time.time()
            m0 = protocols.initial_state(start, fraction.shape, rng)
            M = coarse.solve(fraction, DX, (0, B, 0), m0, A)
            mean_my = float(M[..., 1].sum()) / (geometry.MS * fraction.sum())
            hz = probe_hz(M)
            print(
                f"{name:15s} {start:7s} {mean_my:8.4f} "
                + " ".join(f"{h:10.2f}" for h in hz)
                + f" {time.time() - begin:9.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
