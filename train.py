"""Learn a wall energy E_theta[m] with a CNN on a coarse grid, trained on the equilibria it relaxes to.

The 5 nm data is block-averaged to COARSEN^3-cell coarse cells (renormalised to |m| = 1). On the
coarse grid the effective field is h_demag + h_exchange + h_ext + h_theta, with
h_theta = -dE_theta/dm / (mu0 Ms V) a learned wall energy on top of exchange (E_theta starts at 0,
see model.py). From a coarse stored state m_t, magnum.np's Barzilai-Borwein minimiser at the next
applied field relaxes (without gradients) to an equilibrium m*, which should be the coarse next
stored state m_{t+1}. The gradient of |m* - m_{t+1}|^2 comes from the implicit function theorem at m*:

    dL/dtheta = d/dtheta sum(w . h_theta(m*)),  with  A w = P dL/dm*,
    A v = (m . h_eff) v - P (h_demag(v) + h_exchange(v) + Dh_theta[v])   (the Hessian on the sphere / (mu0 Ms V))

where P projects onto the tangent plane of each cell's m. A is solved by conjugate gradients.
"""

import json
import logging
import math
from pathlib import Path

import magnumnp
import torch
from magnumnp.common.constants import mu_0
from magnumnp.solvers import MinimizerBB
from the_well.data import WellDataset
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

from model import DemagField, ExchangeField, WallEnergy, effective_field
from datagen import A, DM_TOL, DX, N, Ms

# importing magnum.np makes float64 on the GPU torch's default; the model, the loader
# (pin_memory) and the demag kernel need torch's usual float32 on the CPU
torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")
magnumnp.set_log_level(logging.WARNING)

DATASET = "datasets/shield"
RUN_DIR = Path("results/wall_energy")
MAX_RELAX_STEPS = 2000  # MinimizerBB's maxiter; relaxation stops earlier at DM_TOL (as in datagen.py)
N_CG_STEPS = 50  # conjugate-gradient iterations for the adjoint solve A w = P dL/dm*
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
N_STEPS = 5000
WARMUP_STEPS = 200
LOG_EVERY = 50
VALID_EVERY = 500
N_VALID_SAMPLES = 64

KD = 0.5 * mu_0 * Ms**2  # [J/m^3] scale of the learned energy density
COARSEN = 5  # coarse cell = COARSEN^3 data cells; must divide N = (100, 50, 20): 2, 5 or 10
TAU_MIN, TAU_MAX = 1e-13, 1e-5  # MinimizerBB's step-size bounds
# first step size: MinimizerBB starts at TAU_MIN, but in float32 such a step moves no cell at all,
# and the next BB step is then 0/0; 1e-9 reaches the float64 end states (later BB steps: 1e-7..1e-5)
TAU_START = 1e-9
DEVICE = torch.device("cuda")


@torch.no_grad()
def relax(model, demag_field, exchange_field, m: Tensor, h_ext: Tensor) -> tuple[Tensor, Tensor]:
    """magnum.np's MinimizerBB.minimize, batched: m (B, X, Y, Z, 3) -> relaxed m, converged (B,).

    Barzilai-Borwein steps until every sample's max |m x (m x h)| is below DM_TOL [A/m], at most
    MAX_RELAX_STEPS. Unlike MinimizerBB, the first step is TAU_START, and a BB step that is
    undefined (0/0, once a sample has stopped moving in float32) falls back to TAU_START.
    """
    midpoint = MinimizerBB(terms=[])._midpoint  # magnum.np's norm-preserving update
    dims = (1, 2, 3, 4)
    tau = torch.full((m.shape[0], 1, 1, 1), TAU_START, device=m.device)
    h = effective_field(model, demag_field, exchange_field, m, h_ext)
    torque = torch.linalg.cross(m, torch.linalg.cross(m, h))
    for i in range(MAX_RELAX_STEPS):
        converged = torque.abs().amax(dims) < DM_TOL
        if converged.all():
            break
        # samples that have converged stay where they are
        m_new = torch.where(converged[:, None, None, None, None], m, midpoint(m, h, tau))
        h = effective_field(model, demag_field, exchange_field, m_new, h_ext)
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


def tangent(m: Tensor, v: Tensor) -> Tensor:
    """Project v onto the tangent plane of each cell's unit vector m: v - (m . v) m."""
    return v - (m * v).sum(-1, keepdim=True) * m


def solve_adjoint(model, demag_field, exchange_field, m: Tensor, h_ext: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Solve A w = b at the equilibrium m by batched conjugate gradients; returns w and |r| / |b| (B,).

    A v = (m . h_eff) v - P (h_demag(v) + h_exchange(v) + Dh_theta[v]) is the Hessian of the total
    energy on the sphere divided by mu0 Ms V, symmetric and positive definite at a stable equilibrium.
    Demag and exchange are linear in m; Dh_theta[v] is a Hessian-vector product of E_theta
    (Dh_theta is symmetric, so a VJP gives it).
    """
    dims = (1, 2, 3, 4)
    m = m.detach().requires_grad_()
    h_theta = model.field(m)  # keeps its graph in m for the Hessian-vector products
    h_fixed = demag_field(m.detach()) + exchange_field(m.detach()) + h_ext[:, None, None, None, :]
    m_dot_h = (m * (h_fixed + h_theta)).sum(-1, keepdim=True).detach()

    def apply_A(v):
        (dh_theta,) = torch.autograd.grad(h_theta, m, grad_outputs=v, retain_graph=True)
        return tangent(m.detach(), m_dot_h * v - demag_field(v) - exchange_field(v) - dh_theta)

    w, r = torch.zeros_like(b), b.clone()
    p, rr = r.clone(), (r * r).sum(dims, keepdim=True)
    for _ in range(N_CG_STEPS):
        Ap = apply_A(p)
        pAp = (p * Ap).sum(dims, keepdim=True)
        # stop a sample at a direction of non-positive curvature (an unstable equilibrium)
        alpha = torch.where(pAp > 0, rr / pAp, 0.0)
        w, r = w + alpha * p, r - alpha * Ap
        rr_new = (r * r).sum(dims, keepdim=True)
        p = r + torch.where(rr > 0, rr_new / rr, 0.0) * p
        rr = rr_new
    residual = (rr.sqrt() / (b * b).sum(dims, keepdim=True).sqrt().clamp_min(1e-30)).flatten()
    return w, residual


def coarsen(m: Tensor) -> Tensor:
    """Block-average m (B, X, Y, Z, 3) over COARSEN^3 cells and renormalise to |m| = 1."""
    m = F.avg_pool3d(m.permute(0, 4, 1, 2, 3), COARSEN).permute(0, 2, 3, 4, 1)
    return F.normalize(m, dim=-1)


def unpack(batch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Coarse m_t and m_{t+1} (B, X, Y, Z, 3) and the applied field at t+1, h_ext (B, 3) [A/m]."""
    m_start = coarsen(batch["input_fields"][:, 0].to(DEVICE))
    m_target = coarsen(batch["output_fields"][:, 0].to(DEVICE))
    b = batch["output_scalars"][:, 0].to(DEVICE)  # (B, 2): Bx, By [T]
    h_ext = torch.cat([b, torch.zeros_like(b[:, :1])], dim=1) / mu_0
    return m_start, m_target, h_ext


def train_step(model, optimizer, scheduler, demag_field, exchange_field, batch) -> dict[str, Tensor]:
    m_start, m_target, h_ext = unpack(batch)
    m_eq, converged = relax(model, demag_field, exchange_field, m_start, h_ext)
    loss = ((m_eq - m_target) ** 2).sum(-1).mean()

    # implicit function theorem at m*: dL/dtheta = d/dtheta sum(w . h_theta(m*)), A w = P dL/dm*
    # samples that did not reach DM_TOL are not at an equilibrium: the adjoint is meaningless there
    dL_dm = 2 * (m_eq - m_target) / m_eq[..., 0].numel() * converged[:, None, None, None, None]
    w, cg_residual = solve_adjoint(model, demag_field, exchange_field, m_eq, h_ext, tangent(m_eq, dL_dm))
    optimizer.zero_grad()
    (w * model.field(m_eq)).sum().backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # loss_unmoved: the loss if relax() left m_t where it was, the baseline to beat
    loss_unmoved = ((m_start - m_target) ** 2).sum(-1).mean()
    return {
        "loss": loss,
        "loss_unmoved": loss_unmoved,
        "grad_norm": grad_norm.detach(),
        "converged": converged.float().mean(),
        "cg_residual": (cg_residual * converged).sum() / converged.sum().clamp_min(1),
    }


@torch.no_grad()
def evaluate(model, demag_field, exchange_field, loader) -> dict[str, Tensor]:
    model.eval()
    losses, losses_unmoved, converged = [], [], []
    for batch in loader:
        m_start, m_target, h_ext = unpack(batch)
        m_eq, batch_converged = relax(model, demag_field, exchange_field, m_start, h_ext)
        losses.append(((m_eq - m_target) ** 2).sum(-1).mean(dim=(1, 2, 3)))
        losses_unmoved.append(((m_start - m_target) ** 2).sum(-1).mean(dim=(1, 2, 3)))
        converged.append(batch_converged.float())
    model.train()
    return {
        "valid_loss": torch.cat(losses).mean(),
        "valid_loss_unmoved": torch.cat(losses_unmoved).mean(),
        "valid_converged": torch.cat(converged).mean(),
    }


def forever(loader):
    while True:
        yield from loader


def main():
    torch.manual_seed(0)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    train_set = WellDataset(path=DATASET, well_split_name="train")
    valid_set = WellDataset(path=DATASET, well_split_name="valid")
    # evenly spaced (m_t, m_{t+1}) pairs, so validation covers every trajectory and field
    valid_set = Subset(
        valid_set, range(0, len(valid_set), len(valid_set) // N_VALID_SAMPLES)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True
    )

    assert all(n % COARSEN == 0 for n in N), f"COARSEN = {COARSEN} does not divide the mesh {N}"
    n_coarse = [n // COARSEN for n in N]
    dx_coarse = [d * COARSEN for d in DX]
    model = WallEnergy(Kd=KD, V=math.prod(dx_coarse), Ms=Ms).to(DEVICE)
    demag_field = DemagField(n=n_coarse, dx=dx_coarse, Ms=Ms, device=DEVICE)
    exchange_field = ExchangeField(dx=dx_coarse, A=A, Ms=Ms)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_STEPS),
            CosineAnnealingLR(optimizer, T_max=N_STEPS - WARMUP_STEPS),
        ],
        milestones=[WARMUP_STEPS],
    )

    log, window = [], []
    for step, batch in enumerate(forever(train_loader), start=1):
        window.append(train_step(model, optimizer, scheduler, demag_field, exchange_field, batch))

        if step % LOG_EVERY == 0:
            entry = {"step": step, "lr": scheduler.get_last_lr()[0]}
            entry |= {
                name: torch.stack([w[name] for w in window]).mean().item()
                for name in window[0]
            }
            window = []
            if step % VALID_EVERY == 0:
                entry |= {
                    name: value.item()
                    for name, value in evaluate(
                        model, demag_field, exchange_field, valid_loader
                    ).items()
                }
                torch.save(model.state_dict(), RUN_DIR / "model.pt")
            log.append(entry)
            print(entry, flush=True)
            (RUN_DIR / "log.json").write_text(json.dumps(log, indent=1))

        if step == N_STEPS:
            break


if __name__ == "__main__":
    main()
