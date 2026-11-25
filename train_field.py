"""train.py with a learned field instead of a learned energy: the CNN outputs h_theta directly (field_model.py).

Everything else is train.py's: the coarse grid, relax(), the loss and the implicit-function-theorem
gradient dL/dtheta = d/dtheta sum(w . h_theta(m*)). Without an energy behind it, Dh_theta is not
symmetric, so the adjoint system is the transposed one,

    A^T w = P dL/dm*,
    A^T v = (m . h_eff) v - P (h_demag(v) + h_exchange(v) + (Dh_theta)^T v),

(demag and exchange are symmetric), solved by GMRES instead of conjugate gradients. A direct field
has no energy for relax() to minimise, so its Barzilai-Borwein iteration is not guaranteed to
converge: the "converged" fraction in the log shows whether it does.
"""

import json
from pathlib import Path

import torch
from the_well.data import WellDataset
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

import train
from datagen import A, DX, N, Ms
from field_model import LearnedField
from model import DemagField, ExchangeField

RUN_DIR = Path("results/learned_field_pilot_25nm")
N_GMRES_STEPS = 50  # Krylov vectors for the adjoint solve: as many operator calls as train.py's CG
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
N_STEPS = 1500
WARMUP_STEPS = 100
LOG_EVERY = 50
VALID_EVERY = 250
N_VALID_SAMPLES = 64


def gmres(apply, b: Tensor, n_steps: int) -> Tensor:
    """Batched GMRES without restarts: per sample, the x in the n_steps-dimensional Krylov space of
    apply and b that minimises |b - apply(x)|. b (B, ...) -> x (B, ...); apply acts per sample."""
    n_batch = b.shape[0]
    beta = b.flatten(1).norm(dim=1)  # (B,)
    basis = b.new_zeros(n_batch, n_steps + 1, b[0].numel())  # orthonormal Krylov vectors
    hessenberg = b.new_zeros(n_batch, n_steps + 1, n_steps)
    basis[:, 0] = b.flatten(1) / beta.clamp_min(1e-30)[:, None]
    for j in range(n_steps):
        v = apply(basis[:, j].view_as(b)).flatten(1)
        # classical Gram-Schmidt, twice, to keep the basis orthogonal in float32
        for _ in range(2):
            coefficients = torch.einsum("bkn,bn->bk", basis[:, : j + 1], v)
            v = v - torch.einsum("bkn,bk->bn", basis[:, : j + 1], coefficients)
            hessenberg[:, : j + 1, j] += coefficients
        hessenberg[:, j + 1, j] = v.norm(dim=1)
        basis[:, j + 1] = v / v.norm(dim=1).clamp_min(1e-30)[:, None]
    # y = argmin |beta e_1 - H y|; the pseudo-inverse also copes with a singular H (b = 0, breakdown)
    y = beta[:, None] * torch.linalg.pinv(hessenberg)[:, :, 0]  # (B, n_steps)
    return torch.einsum("bkn,bk->bn", basis[:, :n_steps], y).view_as(b)


def solve_adjoint(model, demag_field, exchange_field, m: Tensor, h_ext: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Solve A^T w = b at the equilibrium m by batched GMRES; returns w and |b - A^T w| / |b| (B,).

    A^T v = (m . h_eff) v - P (h_demag(v) + h_exchange(v) + (Dh_theta)^T v) is train.solve_adjoint's
    operator: a vector-Jacobian product gives (Dh_theta)^T v, and demag and exchange are symmetric.
    """
    m = m.detach().requires_grad_()
    h_theta = model.field(m)  # keeps its graph in m for the vector-Jacobian products
    h_fixed = demag_field(m.detach()) + exchange_field(m.detach()) + h_ext[:, None, None, None, :]
    m_dot_h = (m * (h_fixed + h_theta)).sum(-1, keepdim=True).detach()

    # P A^T P, and w projected: once GMRES has converged, Arnoldi normalises rounding noise into new
    # Krylov vectors; without the projections their normal parts reach w through the operator's null
    # space and bias the gradient (30 % of w in the float64 check with 400 Krylov vectors)
    def apply_AT(v):
        v = train.tangent(m.detach(), v)
        (dh_theta_T_v,) = torch.autograd.grad(h_theta, m, grad_outputs=v, retain_graph=True)
        return train.tangent(m.detach(), m_dot_h * v - demag_field(v) - exchange_field(v) - dh_theta_T_v)

    w = train.tangent(m.detach(), gmres(apply_AT, b, N_GMRES_STEPS))
    residual = (b - apply_AT(w)).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1).clamp_min(1e-30)
    return w, residual


def train_step(model, optimizer, scheduler, demag_field, exchange_field, batch) -> dict[str, Tensor]:
    m_start, m_target, h_ext = train.unpack(batch)
    m_eq, converged = train.relax(model, demag_field, exchange_field, m_start, h_ext)
    loss = ((m_eq - m_target) ** 2).sum(-1).mean()

    # implicit function theorem at m*: dL/dtheta = d/dtheta sum(w . h_theta(m*)), A^T w = P dL/dm*
    # samples that did not reach DM_TOL are not at an equilibrium: the adjoint is meaningless there
    dL_dm = 2 * (m_eq - m_target) / m_eq[..., 0].numel() * converged[:, None, None, None, None]
    w, gmres_residual = solve_adjoint(model, demag_field, exchange_field, m_eq, h_ext, train.tangent(m_eq, dL_dm))
    optimizer.zero_grad()
    h_theta = model.field(m_eq)
    (w * h_theta).sum().backward()
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
        "gmres_residual": (gmres_residual * converged).sum() / converged.sum().clamp_min(1),
        "h_theta_rms": h_theta.detach().pow(2).mean().sqrt(),  # [A/m], at the relaxed states
    }


def main():
    torch.manual_seed(0)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    train_set = WellDataset(path=train.DATASET, well_split_name="train")
    valid_set = WellDataset(path=train.DATASET, well_split_name="valid")
    # evenly spaced (m_t, m_{t+1}) pairs, so validation covers every trajectory and field
    valid_set = Subset(valid_set, range(0, len(valid_set), len(valid_set) // N_VALID_SAMPLES))
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

    n_coarse = [n // train.COARSEN for n in N]
    dx_coarse = [d * train.COARSEN for d in DX]
    model = LearnedField(Ms=Ms).to(train.DEVICE)
    demag_field = DemagField(n=n_coarse, dx=dx_coarse, Ms=Ms, device=train.DEVICE)
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
    for step, batch in enumerate(train.forever(train_loader), start=1):
        window.append(train_step(model, optimizer, scheduler, demag_field, exchange_field, batch))

        if step % LOG_EVERY == 0:
            entry = {"step": step, "lr": scheduler.get_last_lr()[0]}
            entry |= {name: torch.stack([w[name] for w in window]).mean().item() for name in window[0]}
            window = []
            if step % VALID_EVERY == 0:
                valid = train.evaluate(model, demag_field, exchange_field, valid_loader)
                entry |= {name: value.item() for name, value in valid.items()}
                torch.save(model.state_dict(), RUN_DIR / "model.pt")
            log.append(entry)
            print(entry, flush=True)
            (RUN_DIR / "log.json").write_text(json.dumps(log, indent=1))

        if step == N_STEPS:
            break


if __name__ == "__main__":
    main()
