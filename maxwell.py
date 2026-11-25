"""Maxwell M(H) baseline: the block as a soft magnet with a single-valued law, without exchange.

Every cell follows the saturating law of Shield_Optimization_DS.pdf, eq. 1,

    m = x / sqrt(1 + |x|^2),    x = chi H / Ms,

in the total field H = h_ext + h_demag(m), with magnum.np's FFT demag field on the cell grid N,
DX. There are no domains and no history, so the equilibrium is unique. It is the fixed point of

    b = m + H / Ms                          (the flux density mu0 (H + M), in units of mu0 Ms)
    m = the magnetisation the law gives for b

Solving the law for m at fixed b keeps |m| < 1 and copes with the sharp knee of chi = 1e5, where
Newton's method on H stalls. One iteration is a proximal-gradient step of size 1 on the convex
magnetostatic energy, so it converges from any start; momentum (FISTA with restarts) makes it fast.

Runs in magnum.np's defaults (float64 on the GPU, as datagen.py): H inside the block is ~1e-5 of
h_ext, which float32 cannot resolve.
"""

import math
import time

import numpy as np
import torch
from magnumnp.common.constants import mu_0
from magnumnp.field_terms import DemagField
from torch import Tensor
from torch.nn import functional as F

from datagen import (
    Hmax,
    Hmin,
    Ms,
    chi,
    loop_from_saturated,
    material,
    setup,
    steps,
)

# cell grid of the sweep in main(): the magnet of datasets/shield (0.5 x 0.25 x 0.1 um); without
# exchange only its proportions matter, not its size
N = [100, 50, 20]
DX = [5e-9] * 3
AXIS = np.array([1.0, 0.0, 0.0])  # field axis of the sweep in main()
# converged when, in every cell, the total field is within H_TOL of the field the law needs for m;
# the scale to stay far below is the knee of the law, Ms / chi = 8 A/m
H_TOL = 1e-3 * Ms / chi  # [A/m]
MAX_ITERATIONS = 20_000
N_LAW_STEPS = 20  # Newton steps of the per-cell law; 15 reach round-off at chi = 1e5


def magnetisation(b: Tensor) -> Tensor:
    """The law solved for b = m + H / Ms: b (X, Y, Z, 3) -> m (X, Y, Z, 3) in units of Ms, |m| < 1."""
    norm = b.norm(dim=-1, keepdim=True)
    # with t = chi |H| / Ms the law reads |b| = f(t) = t / sqrt(1 + t^2) + t / chi; f rises and is
    # concave, so Newton's method from t = 0 climbs to the root without overshooting
    t = torch.zeros_like(norm)
    for _ in range(N_LAW_STEPS):
        f = t / torch.sqrt(1 + t**2) + t / chi
        df = (1 + t**2) ** -1.5 + 1 / chi
        t = t + (norm - f) / df
    return t / torch.sqrt(1 + t**2) * F.normalize(b, dim=-1)


def solve(state, demag, h_ext: Tensor) -> Tensor:
    """Equilibrium m (X, Y, Z, 3), in units of Ms, in the applied field h_ext (X, Y, Z, 3) [A/m]."""
    m = torch.zeros_like(h_ext)
    m_ahead, theta = (
        m,
        1.0,
    )  # m_ahead: m moved on by the momentum, where each iteration starts
    for _ in range(MAX_ITERATIONS):
        state.m = m_ahead
        H = h_ext + demag.h(state)
        m_new = magnetisation(m_ahead + H / Ms)
        # (m_new - m_ahead) Ms is H minus the field the law needs for m_new
        if (m_new - m_ahead).abs().max() * Ms < H_TOL:
            return m_new
        if (
            (m_ahead - m_new) * (m_new - m)
        ).sum() > 0:  # the momentum points uphill: restart it
            theta = 1.0
        theta_new = (1 + math.sqrt(1 + 4 * theta**2)) / 2
        m_ahead = m_new + (theta - 1) / theta_new * (m_new - m)
        m, theta = m_new, theta_new
    raise RuntimeError(
        f"not converged to {H_TOL:.1e} A/m in {MAX_ITERATIONS} iterations"
    )


def main():
    state, _ = setup(n=N, dx=DX, material=material, m0=[0.0, 0.0, 0.0])
    demag = DemagField()

    # the Hmax -> Hmin half of datagen.py's loop: with no history, the way back is the same curve
    field_values = loop_from_saturated(Hmin=Hmin, Hmax=Hmax, steps=steps, axis=AXIS)[
        : steps // 2 + 1
    ]
    start = time.time()
    print("B (mT)   <m> along the field axis")
    for field in field_values:
        m = solve(state, demag, state.Constant(field))
        mean_m = m.mean(dim=(0, 1, 2)).cpu().numpy()
        print(f"{1e3 * mu_0 * field @ AXIS:7.1f}  {mean_m @ AXIS:8.5f}", flush=True)
    print(f"{len(field_values)} fields in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
