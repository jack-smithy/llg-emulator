"""Coarse-grid micromagnetics with a batch dimension: demag, exchange, the effective field with a learned
correction, Barzilai-Borwein relaxation and the softest curvature of the energy on the sphere.

All fields are in A/m; m is (B, X, Y, Z, 3) with |m| = 1 per cell; the applied field h_ext is (B, 3).
mask (B, X, Y, Z) marks the magnet in grids padded with air (data.StoredStates: m = 0 outside), or is
None for no padding. Padded cells have no exchange with the magnet, no field and do not move, so a
padded magnet behaves exactly as it does alone on its own grid.
"""

import logging

import magnumnp
import torch
from magnumnp.common import Mesh, State
from magnumnp.common.constants import mu_0
from magnumnp.field_terms import DemagField as _DemagField
from torch import Tensor, nn

from datagen import Ms

# importing magnum.np makes float64 on the GPU torch's default; the model, the loader (pin_memory)
# and the demag kernel need torch's usual float32 on the CPU
torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")
# relaxing to DM_TOL needs the field to ~1e-5 of its size; TF32 convolutions (which cuDNN may pick,
# e.g. at batch size 1) give ~1e-3: one test loop at a time then converged in 81 % of the steps, not 100 %
torch.backends.cudnn.allow_tf32 = False
magnumnp.set_log_level(logging.WARNING)

DEVICE = torch.device("cuda")
KD = 0.5 * mu_0 * Ms**2  # [J/m^3] scale of the learned energy density
DM_TOL = 10  # [A/m] torque tolerance of the relaxation, as datagen.py's --dm-tol
MAX_RELAX_STEPS = 2000  # relaxation stops earlier at DM_TOL
TAU_MIN, TAU_MAX = 1e-13, 1e-5  # MinimizerBB's step-size bounds
# first step size: MinimizerBB starts at TAU_MIN, but in float32 such a step moves no cell at all, and
# the next BB step is then 0/0; 1e-9 reaches the float64 end states (later BB steps: 1e-7..1e-5)
TAU_START = 1e-9
N_LANCZOS_STEPS = 40  # Krylov dimension for the softest direction (exact to a few % when it is negative)


class DemagField(nn.Module):
    """magnum.np's DemagField.h with a batch dimension, open boundaries (zero padding)."""

    def __init__(self, n, dx, Ms, device):
        super().__init__()
        kernel = _DemagField()._init_N(State(Mesh(n=n, dx=dx)))
        self.kernel = [[component.to(device) for component in row] for row in kernel]
        self.Ms = Ms

    def forward(self, m: Tensor) -> Tensor:
        nx, ny, nz = m.shape[1:4]
        padded = (2 * nx, 2 * ny, 2 * nz)
        m_fft = [
            torch.fft.rfftn(self.Ms * m[..., i], s=padded, dim=(1, 2, 3))
            for i in range(3)
        ]
        h = []
        for row in self.kernel:
            h_fft = row[0] * m_fft[0] + row[1] * m_fft[1] + row[2] * m_fft[2]
            h.append(torch.fft.irfftn(h_fft, s=padded, dim=(1, 2, 3))[:, :nx, :ny, :nz])
        return torch.stack(h, dim=-1)


class ExchangeField(nn.Module):
    """magnum.np's ExchangeField.h for uniform A and Ms, open boundaries: a missing neighbour at the
    surface, or one in the padding, adds nothing."""

    def __init__(self, dx, A, Ms):
        super().__init__()
        self.dx = dx
        self.A = A
        self.Ms = Ms

    def forward(self, m: Tensor, mask: Tensor | None = None) -> Tensor:
        dx, dy, dz = self.dx
        h = torch.zeros_like(m)
        diff = (m[:, 1:] - m[:, :-1]) / dx**2
        if mask is not None:  # only pairs of magnet cells
            diff = diff * (mask[:, 1:] * mask[:, :-1])[..., None]
        h[:, :-1] += diff
        h[:, 1:] -= diff
        diff = (m[:, :, 1:] - m[:, :, :-1]) / dy**2
        if mask is not None:
            diff = diff * (mask[:, :, 1:] * mask[:, :, :-1])[..., None]
        h[:, :, :-1] += diff
        h[:, :, 1:] -= diff
        diff = (m[:, :, :, 1:] - m[:, :, :, :-1]) / dz**2
        if mask is not None:
            diff = diff * (mask[:, :, :, 1:] * mask[:, :, :, :-1])[..., None]
        h[:, :, :, :-1] += diff
        h[:, :, :, 1:] -= diff
        return 2 * self.A / (mu_0 * self.Ms) * h


class NoCorrection(nn.Module):
    """Plain coarse physics: h_theta = 0."""

    def field(self, m: Tensor, mask: Tensor | None = None) -> Tensor:
        return torch.zeros_like(m)


def effective_field(
    model,
    demag_field,
    exchange_field,
    m: Tensor,
    h_ext: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """h_demag + h_exchange + h_ext + h_theta in the magnet, 0 in the padding: (B, X, Y, Z, 3)."""
    h = (
        demag_field(m)
        + exchange_field(m, mask)
        + h_ext[:, None, None, None, :]
        + model.field(m, mask)
    )
    return h if mask is None else h * mask[..., None]


def tangent(m: Tensor, v: Tensor) -> Tensor:
    """Project v onto the tangent plane of each cell's unit vector m: v - (m . v) m."""
    return v - (m * v).sum(-1, keepdim=True) * m


def midpoint(m, h, tau):
    """
    Solving the the semi-implicit midpoint scheme:
        m_i+1 = m_i + tau * (m_i + m_i+1)/2 x (m_i x Heff[m_i])

    see "Abert, 'Efficient Energyminimization in Finite-Difference Micromagnetics', 2014"
    see "Goldfarb, 'A Curvilinear Search Method for p-Harmonic Flows on Spheres', 2009"

    Copied from magnum.np `MinimizerBB._midpoint`
    """
    mxh = torch.linalg.cross(m, h)
    mx, my, mz = m.unbind(-1)
    mxh_x, mxh_y, mxh_z = mxh.unbind(-1)

    N = 4 + tau * tau * (mxh_x * mxh_x + mxh_y * mxh_y + mxh_z * mxh_z)
    return torch.stack(
        [
            (
                4 * mx
                + 4 * tau * (mxh_y * mz - mxh_z * my)
                + tau * tau * mx * (+mxh_x * mxh_x - mxh_y * mxh_y - mxh_z * mxh_z)
                + 2 * tau * tau * mxh_x * (mxh_y * my + mxh_z * mz)
            )
            / N,
            (
                4 * my
                + 4 * tau * (mxh_z * mx - mxh_x * mz)
                + tau * tau * my * (-mxh_x * mxh_x + mxh_y * mxh_y - mxh_z * mxh_z)
                + 2 * tau * tau * mxh_y * (mxh_z * mz + mxh_x * mx)
            )
            / N,
            (
                4 * mz
                + 4 * tau * (mxh_x * my - mxh_y * mx)
                + tau * tau * mz * (-mxh_x * mxh_x - mxh_y * mxh_y + mxh_z * mxh_z)
                + 2 * tau * tau * mxh_z * (mxh_x * mx + mxh_y * my)
            )
            / N,
        ],
        dim=-1,
    )
    # mumax code
    # // m = 1 / (4 + τ²(m x H)²) [{4 - τ²(m x H)²} m - 4τ(m x m x H)]
    # // note: torque from LLNoPrecess has negative sign


@torch.no_grad()
def relax(
    model,
    demag_field,
    exchange_field,
    m: Tensor,
    h_ext: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """magnum.np's MinimizerBB.minimize, batched: relaxed m and converged (B,).

    Barzilai-Borwein steps until every sample's max |m x (m x h)| is below DM_TOL, at most
    MAX_RELAX_STEPS. Unlike MinimizerBB, the first step is TAU_START, and a BB step that is undefined
    (0/0, once a sample has stopped moving in float32) falls back to TAU_START.
    """
    dims = (1, 2, 3, 4)
    tau = torch.full((m.shape[0], 1, 1, 1), TAU_START, device=m.device)
    # padded cells: m = 0 and h = 0, so no torque, and the update leaves them at 0
    h = effective_field(model, demag_field, exchange_field, m, h_ext, mask)
    torque = torch.linalg.cross(m, torch.linalg.cross(m, h))
    for i in range(MAX_RELAX_STEPS):
        converged = torque.abs().amax(dims) < DM_TOL
        if converged.all():
            break
        # samples that have converged stay where they are
        m_new = torch.where(
            converged[:, None, None, None, None], m, midpoint(m, h, tau)
        )
        h = effective_field(model, demag_field, exchange_field, m_new, h_ext, mask)
        torque_new = torch.linalg.cross(m_new, torch.linalg.cross(m_new, h))
        s, y = m_new - m, torque_new - torque
        if i % 2 == 0:
            tau = (s * s).sum(dims) / (s * y).sum(dims)
        else:
            tau = (s * y).sum(dims) / (y * y).sum(dims)
        tau = torch.where(torch.isfinite(tau), tau, TAU_START)
        tau = tau.abs().clamp(TAU_MIN, TAU_MAX)[:, None, None, None]
        m, torque = m_new, torque_new
    return m, torque.abs().amax(dims) < DM_TOL


def lanczos_softest(apply_A, v0: Tensor, n_steps: int) -> Tensor:
    """Ritz vector of the smallest eigenvalue of the symmetric operator apply_A, per sample, from the
    n_steps Krylov space of v0 (Lanczos with full reorthogonalisation): v0 (B, ...) -> (B, ...)."""
    basis = v0.new_zeros(v0.shape[0], n_steps, v0[0].numel())
    projected = v0.new_zeros(v0.shape[0], n_steps, n_steps)  # Q^T A Q, upper triangle
    q = v0.flatten(1)
    for j in range(n_steps):
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-30)
        basis[:, j] = q
        w = apply_A(q.view_as(v0)).flatten(1)
        projected[:, : j + 1, j] = torch.einsum("bkn,bn->bk", basis[:, : j + 1], w)
        for _ in range(2):  # full reorthogonalisation, twice for float32
            coefficients = torch.einsum("bkn,bn->bk", basis[:, : j + 1], w)
            w = w - torch.einsum("bkn,bk->bn", basis[:, : j + 1], coefficients)
        q = w
    projected = (
        projected
        + projected.transpose(1, 2)
        - torch.diag_embed(projected.diagonal(dim1=1, dim2=2))
    )
    _, ritz = torch.linalg.eigh(projected)  # eigenvalues ascending
    return torch.einsum("bkn,bk->bn", basis, ritz[:, :, 0]).view_as(v0)


def softest_curvature(
    model,
    demag_field,
    exchange_field,
    m: Tensor,
    h_ext: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """kappa (B,) [A/m]: the curvature v.Av / v.v of the coarse energy on the sphere (divided by
    mu0 Ms V) along its softest tangent direction v at the states m, with

        A v = (m . h_eff) v - P (h_demag(v) + h_exchange(v) + Dh_theta[v]),   P = 1 - m m^T.

    Negative if a state is not a minimum. v comes from Lanczos without gradients; kappa is
    differentiable in theta when gradients are on. Tangent vectors are 0 in the padding."""
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        m_fixed = m.detach()
        m = m_fixed.clone().requires_grad_()
        h_theta = model.field(m, mask)  # graph in m (and in theta when training)
        h_fixed = (
            demag_field(m_fixed)
            + exchange_field(m_fixed, mask)
            + h_ext[:, None, None, None, :]
        )
        m_dot_h = (m_fixed * (h_fixed + h_theta)).sum(-1, keepdim=True)
        # the tangent planes, and 0 in the padding (where m = 0 and the projection would keep all of v)
        project = (
            (lambda v: tangent(m_fixed, v))
            if mask is None
            else (lambda v: tangent(m_fixed, v) * mask[..., None])
        )

        def apply_A(v, keep_graph):
            v = project(v)
            (dh_theta,) = torch.autograd.grad(
                h_theta, m, grad_outputs=v, retain_graph=True, create_graph=keep_graph
            )
            m_dot_h_v = m_dot_h if keep_graph else m_dot_h.detach()
            return project(
                m_dot_h_v * v - demag_field(v) - exchange_field(v, mask) - dh_theta
            )

        v0 = project(torch.randn_like(m_fixed))
        v = project(
            lanczos_softest(lambda v: apply_A(v, False), v0, N_LANCZOS_STEPS)
        ).detach()
        dims = (1, 2, 3, 4)
        kappa = (v * apply_A(v, create_graph)).sum(dims) / (v * v).sum(dims)
    return kappa if create_graph else kappa.detach()
