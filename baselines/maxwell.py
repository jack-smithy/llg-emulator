"""Nonlinear magnetostatics ("Maxwell solver"), the PDF's continuum baseline.

Every cell follows the saturating law of the PDF, eq. 1,

    M = law(H) = Ms x / sqrt(1 + x^2) along H,    x = chi |H| / Ms,

where H = H_ext + H_demag(M) is the total field. There is no exchange, so no domains and no
history: the solution is unique. We solve for the total field, R(H) = H - H_ext - H_demag(law(H))
= 0, by Newton's method. law never reaches |M| = Ms, so no step can push a cell past saturation.

With u = law' dH (the change of M), the Newton system becomes (law'^-1 - D) u = -R, D being the
demag operator. law is the gradient of a convex function and -D is positive semidefinite, so this
matrix is symmetric positive definite and scipy's preconditioned CG solves it (float64); then
dH = law'^-1 u. A CG solution with a small relative residual always lowers |R|, and a backtracking
line search on |R| keeps every step. As in the PDF, chi is stepped up geometrically from 1, each
solve starting from the last.

A cell partly filled with magnet (volume fraction f) holds the cell-averaged M, which follows the
same law with Ms -> f Ms and chi -> f chi.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy import constants
from scipy.sparse.linalg import LinearOperator, cg

from shield import strayfield
from shield.geometry import MS

CHI = 1e5  # the PDF's initial susceptibility
CHI_STEPS = 11  # geometric steps from chi = 1, as in the PDF
TOLERANCE = 1e-8  # converged when max |R| < TOLERANCE |H_ext|
MAX_NEWTON = 100
CG_TOLERANCE = 1e-3  # relative residual of each Newton system; the next Newton step corrects the rest
MAX_CG = 20_000
SELF_DEMAG = 1 / 3  # a cubic cell's own demagnetising factor, used in the preconditioner


def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cell-wise dot product of vector fields (..., 3), keeping the last axis: (..., 1)."""
    return (a * b).sum(-1, keepdim=True)


def law(H: torch.Tensor, ms: torch.Tensor, chi_cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """M = law(H) per cell (the PDF's eq. 1) and s = 1 / sqrt(1 + x^2) = sqrt(1 - |M|^2 / ms^2)."""
    s = 1 / torch.sqrt(1 + chi_cell**2 * dot(H, H) / ms**2)
    return chi_cell * s * H, s


def as_scipy_operator(
    apply: Callable[[torch.Tensor], torch.Tensor], like: torch.Tensor
) -> LinearOperator:
    """A torch operator on fields shaped like `like`, as a scipy operator on flat numpy vectors."""

    def matvec(v):
        field = torch.as_tensor(v, device=like.device).reshape(like.shape)
        return apply(field).cpu().numpy().ravel()

    return LinearOperator((like.numel(), like.numel()), matvec=matvec, dtype=np.float64)


def solve(fraction: np.ndarray, dx: float, b_ext: Sequence[float], chi: float = CHI) -> torch.Tensor:
    """Equilibrium magnetisation M (nx, ny, nz, 3) in A/m.

    fraction: magnetic volume fraction (nx, ny, nz) on cubic cells of size dx (m).
    b_ext: uniform applied field mu0 H_ext (3,) in tesla.
    """
    demag = strayfield.stray_field_operator(fraction.shape, dx, pad=(0, 0, 0))
    volume = torch.as_tensor(fraction, device=torch.get_default_device())[..., None]  # (nx, ny, nz, 1)
    magnetic = volume > 0
    h_ext = torch.as_tensor(b_ext, dtype=volume.dtype, device=volume.device) / constants.mu_0
    tolerance = TOLERANCE * float(h_ext.norm())
    ms = torch.where(magnetic, MS * volume, MS)  # dummy value in air, where M stays 0
    H = magnetic * h_ext  # (nx, ny, nz, 3), the field in the magnet

    def masked_demag(v):
        return magnetic * demag(v)

    for chi_step in np.geomspace(1, chi, CHI_STEPS):
        chi_cell = torch.where(magnetic, chi_step * volume, chi_step)
        n_cg = 0
        for n_newton in range(MAX_NEWTON):
            H, iterations = newton_step(H, magnetic, ms, chi_cell, h_ext, masked_demag, tolerance)
            if iterations == 0:  # converged
                break
            n_cg += iterations
        else:
            raise RuntimeError(f"Newton did not converge at chi = {chi_step:.3g}")
        print(f"chi {chi_step:9.3g}: {n_newton} Newton steps, {n_cg} CG iterations", flush=True)

    return magnetic * law(H, ms, chi_cell)[0]


def newton_step(
    H: torch.Tensor,
    magnetic: torch.Tensor,
    ms: torch.Tensor,
    chi_cell: torch.Tensor,
    h_ext: torch.Tensor,
    demag: Callable[[torch.Tensor], torch.Tensor],
    tolerance: float,
) -> tuple[torch.Tensor, int]:
    """One Newton step on R(H); returns the new H and the CG iterations, or (H, 0) if converged.

    magnetic: (nx, ny, nz, 1) mask of cells holding magnet; air cells keep M = 0.
    demag: the demag field restricted to the magnet (zero in air cells).
    """
    M, s = law(H, ms, chi_cell)
    M = magnetic * M
    residual = magnetic * (H - h_ext) - demag(M)
    if residual.abs().max() <= tolerance:  # <=: in zero field both are 0
        return H, 0

    # law'^-1 per cell is a v + b M (M . v)
    a = 1 / (chi_cell * s)
    b = 1 / (chi_cell * ms**2 * s**3)

    def inverse_law_jacobian(v):
        return magnetic * (a * v + b * M * dot(M, v))

    def newton_matrix(u):
        return inverse_law_jacobian(u) - demag(u)

    def preconditioner(r):
        # inverse of each cell's own block, (a + 1/3) I + b M M^T (Sherman-Morrison)
        c = a + SELF_DEMAG
        return magnetic * (r - b * M * dot(M, r) / (c + b * dot(M, M))) / c

    iterations = 0

    def count(_):
        nonlocal iterations
        iterations += 1

    # stopping at MAX_CG is fine: a partial CG solution still lowers |R|
    u, _ = cg(
        as_scipy_operator(newton_matrix, H),
        -residual.cpu().numpy().ravel(),
        rtol=CG_TOLERANCE,
        maxiter=MAX_CG,
        M=as_scipy_operator(preconditioner, H),
        callback=count,
    )
    step = inverse_law_jacobian(torch.as_tensor(u, device=H.device).reshape(H.shape))

    norm = residual.norm()
    alpha = 1.0
    while True:  # backtracking line search on |R|
        H_new = H + alpha * step
        M_new = magnetic * law(H_new, ms, chi_cell)[0]
        if (magnetic * (H_new - h_ext) - demag(M_new)).norm() <= (1 - 1e-4 * alpha) * norm:
            return H_new, iterations
        alpha /= 2
        if alpha < 1e-12:
            raise RuntimeError("line search failed")
