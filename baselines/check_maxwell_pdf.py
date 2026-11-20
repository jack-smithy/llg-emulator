"""Reproduce the PDF's Maxwell-solver numbers for the 100 x 50 x 20 um NiFe shield.

PDF, chi = 1e5, B = 50 mT along y:
  - <m_y> = 0.228 (Table 3, 2.5 um cells) and 0.2302 (odd grid, Fig. 6);
  - |mu0 H_z| = 64.5 mT at (0, 22, 12) um (Table 3), i.e. eta = |mu0 H_z| / B = 1.29 ("~1.28" in the text).
Fig. 5: eta stays at ~1.28-1.29 up to ~100 mT for chi = 1000 and 1e4 and drops at 175-200 mT;
chi = 100 gives ~1.24.

Run as a job, e.g.
  sbatch -t 1:00:00 --gres=gpu:1g.24gb:1 --wrap "gto-memcap 8G && uv run python -m baselines.check_maxwell_pdf"
"""

import time

import numpy as np
from scipy import constants

from baselines import maxwell
from shield import geometry, strayfield

SIZE = (100e-6, 50e-6, 20e-6)
B_OPERATE = 0.05  # T, along y
CELL_SIZES = (2.5e-6, 1e-6, 0.5e-6)
SWEEP_CELL = 1e-6
SWEEP_FIELDS = (0.025, 0.05, 0.1, 0.15, 0.2)
SWEEP_CHIS = (100, 1000, 1e4)
# rows of strayfield.probe_points: (0, +22, +12), (0, -22, +12) and (0, +22, +20) um
PROBE_22_12, PROBE_M22_12, PROBE_22_20 = 0, 1, 4


def run(dx: float, b: float, chi: float) -> tuple[float, np.ndarray]:
    """<m_y> and the total field mu0 H (16, 3) in tesla at the probes, for a field b along y."""
    fraction = geometry.volume_fraction("prism", SIZE, dx)
    M = maxwell.solve(fraction, dx, (0, b, 0), chi)
    n = fraction.shape
    H = strayfield.stray_field_operator(n, dx, strayfield.probe_padding(n))(M)
    H_probes = strayfield.sample(H, dx, strayfield.probe_points(SIZE))
    b_probes = constants.mu_0 * H_probes + np.array([0, b, 0])
    mean_my = float(M[..., 1].sum()) / (geometry.MS * fraction.sum())
    return mean_my, b_probes


def main() -> None:
    rows = []
    for dx in CELL_SIZES:
        start = time.time()
        mean_my, b_probes = run(dx, B_OPERATE, maxwell.CHI)
        hz = 1e3 * b_probes[:, 2]  # mT
        eta = abs(hz[PROBE_22_12]) / (1e3 * B_OPERATE)
        rows.append(
            f"{dx * 1e6:6.2f} {mean_my:8.4f} {hz[PROBE_22_12]:10.2f} {hz[PROBE_M22_12]:10.2f} "
            f"{hz[PROBE_22_20]:10.2f} {eta:7.3f} {time.time() - start:9.1f}"
        )
    print("\nchi = 1e5, B = 50 mT along y. PDF: <m_y> 0.228-0.230, Hz(0,22,12) 64.5 mT, eta 1.28-1.29")
    print("dx(um)    <m_y>  Hz(22,12) Hz(-22,12)  Hz(22,20)     eta  time (s)")
    print("\n".join(rows))

    sweep = {}
    for chi in SWEEP_CHIS:
        sweep[chi] = []
        for b in SWEEP_FIELDS:
            _, b_probes = run(SWEEP_CELL, b, chi)
            sweep[chi].append(abs(b_probes[PROBE_22_12, 2]) / b)
    print(f"\neta at (0, 22, 12) um on {SWEEP_CELL * 1e6:g} um cells. PDF Fig. 5: ~1.24 (chi 100), ~1.29 (chi >= 1000)")
    print("B (mT)   " + "".join(f"chi={chi:<8g}" for chi in SWEEP_CHIS))
    for i, b in enumerate(SWEEP_FIELDS):
        print(f"{b * 1e3:6.0f}   " + "".join(f"{sweep[chi][i]:<12.3f}" for chi in SWEEP_CHIS))


if __name__ == "__main__":
    main()
