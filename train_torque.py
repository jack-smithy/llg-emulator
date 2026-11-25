"""Learn a coarse-grid field correction h_theta by making the coarse data states equilibria.

Every stored state is a relaxed (equilibrium) state of the 5 nm physics at its applied field. On the
coarse grid (train.coarsen), coarse demag + coarse exchange + the applied field leave a torque on the
block-averaged state; the learned correction h_theta is trained to cancel it. With ENERGY it is the
field of a learned wall energy, h_theta = -dE_theta/dm / (mu0 Ms V) (model.WallEnergy); otherwise the
network outputs h_theta directly (field_model.LearnedField):

    L = mean over cells |P (h_demag + h_exchange + h_ext + h_theta)(m)|^2 / Ms^2,   P = 1 - m m^T

With ENERGY, a second term asks every data state to be a minimum of the coarse energy, not only an
equilibrium: kappa, the curvature v.Av / v.v of the energy on the sphere along its softest tangent
direction v (A as in train.solve_adjoint, in A/m), must not be negative,

    L_stability = mean relu(-kappa / Ms)^2,

with v found by Lanczos (no gradients) and kappa differentiable in theta. Only negative curvature is
penalised: states just before a switching event are barely stable, and must stay so.

No relaxation is needed to train. Validation also relaxes from m_t at the next field (train.relax)
and compares with m_{t+1}, as train.py does: that is the number that matters.
"""

import json
import math
from pathlib import Path

import torch
from the_well.data import WellDataset
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

import train
from datagen import A, Ms
from field_model import LearnedField
from model import DemagField, ExchangeField, WallEnergy, effective_field

RUN_DIR = Path("results/torque_25nm")
ENERGY = (
    False  # learn a wall energy (h_theta = -dE_theta/dm) instead of the field itself
)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
N_STEPS = 3000
WARMUP_STEPS = 100
LOG_EVERY = 100
VALID_EVERY = 500
N_VALID_SAMPLES = 64
SEED = 0  # torch.manual_seed: initial weights and the order of the training samples
N_TRAIN_TRAJECTORIES = None  # train on a fixed subset of this many train trajectories (None: all)
STABILITY_WEIGHT = 0.0  # weight of L_stability (needs ENERGY: only an energy has a symmetric Hessian)
N_LANCZOS_STEPS = 40  # Krylov dimension for the softest direction (exact to a few % when it is negative)


def make_model(dx_coarse: list[float]):
    """The learned correction on cells of size dx_coarse, starting at h_theta = 0."""
    if ENERGY:
        return WallEnergy(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms).to(train.DEVICE)
    return LearnedField(Ms=Ms).to(train.DEVICE)


def torque_loss(model, demag_field, exchange_field, m: Tensor, h_ext: Tensor) -> Tensor:
    """Mean |P h_eff|^2 / Ms^2 over cells, per sample: m (B, X, Y, Z, 3), h_ext (B, 3) -> (B,)."""
    h = effective_field(model, demag_field, exchange_field, m, h_ext)
    return (train.tangent(m, h) ** 2).sum(-1).mean(dim=(1, 2, 3)) / Ms**2


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
    projected = projected + projected.transpose(1, 2) - torch.diag_embed(projected.diagonal(dim1=1, dim2=2))
    _, ritz = torch.linalg.eigh(projected)  # eigenvalues ascending
    return torch.einsum("bkn,bk->bn", basis, ritz[:, :, 0]).view_as(v0)


def softest_curvature(model, demag_field, exchange_field, m: Tensor, h_ext: Tensor) -> Tensor:
    """kappa (B,) [A/m]: the curvature v.Av / v.v of the total coarse energy on the sphere (divided by
    mu0 Ms V) along its softest tangent direction v at the states m. Negative if a state is not a
    minimum. Differentiable in theta when gradients are on."""
    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        m_fixed = m.detach()
        m = m_fixed.clone().requires_grad_()
        h_theta = model.field(m)  # graph in m (and in theta when training)
        h_fixed = demag_field(m_fixed) + exchange_field(m_fixed) + h_ext[:, None, None, None, :]
        m_dot_h = (m_fixed * (h_fixed + h_theta)).sum(-1, keepdim=True)

        def apply_A(v, keep_graph):
            v = train.tangent(m_fixed, v)
            (dh_theta,) = torch.autograd.grad(h_theta, m, grad_outputs=v, retain_graph=True, create_graph=keep_graph)
            m_dot_h_v = m_dot_h if keep_graph else m_dot_h.detach()
            return train.tangent(m_fixed, m_dot_h_v * v - demag_field(v) - exchange_field(v) - dh_theta)

        v0 = train.tangent(m_fixed, torch.randn_like(m_fixed))
        v = train.tangent(m_fixed, lanczos_softest(lambda v: apply_A(v, False), v0, N_LANCZOS_STEPS)).detach()
        dims = (1, 2, 3, 4)
        kappa = (v * apply_A(v, create_graph)).sum(dims) / (v * v).sum(dims)
    return kappa if create_graph else kappa.detach()


def train_step(
    model, optimizer, scheduler, demag_field, exchange_field, batch
) -> dict[str, Tensor]:
    _, m, h_ext = train.unpack(
        batch
    )  # a stored state m_{t+1} and its own applied field
    torque = torque_loss(model, demag_field, exchange_field, m, h_ext).mean()
    loss = torque
    losses = {"torque_loss": torque.detach()}
    if ENERGY:  # only an energy has a symmetric Hessian, so only then is kappa a curvature
        kappa = softest_curvature(model, demag_field, exchange_field, m, h_ext) / Ms
        instability = (torch.relu(-kappa) ** 2).mean()
        loss = loss + STABILITY_WEIGHT * instability
        losses |= {"instability": instability.detach(), "unstable_fraction": (kappa < 0).float().mean()}

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return losses | {"grad_norm": grad_norm.detach()}


def evaluate(model, demag_field, exchange_field, loader) -> dict[str, Tensor]:
    """train.evaluate (relaxed loss) plus the torque loss (and with ENERGY the softest curvature) on
    the stored valid states."""
    torques, kappas = [], []
    for batch in loader:
        _, m, h_ext = train.unpack(batch)
        with torch.no_grad():
            torques.append(torque_loss(model, demag_field, exchange_field, m, h_ext))
            if ENERGY:
                kappas.append(softest_curvature(model, demag_field, exchange_field, m, h_ext) / Ms)
    result = {"valid_torque_loss": torch.cat(torques).mean()}
    if ENERGY:
        kappa = torch.cat(kappas)
        result |= {"valid_unstable_fraction": (kappa < 0).float().mean(), "valid_kappa_median": kappa.median()}
    return result | train.evaluate(model, demag_field, exchange_field, loader)


def main():
    torch.manual_seed(SEED)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    train_set = WellDataset(path=train.DATASET, well_split_name="train")
    if N_TRAIN_TRAJECTORIES is not None:
        # not the_well's restrict_num_trajectories: in the_well 1.2.0 it keeps only trajectory 0 (every
        # sampled trajectory adds windows 0..W-1 instead of its own windows)
        assert sum(train_set.n_trajectories_per_file) == train_set.n_files  # one trajectory per file
        order = torch.randperm(train_set.n_files, generator=torch.Generator().manual_seed(0))
        trajectories = sorted(order[:N_TRAIN_TRAJECTORIES].tolist())  # nested: 6 of 12 of 24
        # all windows of each chosen file (loops from a random start have fewer than from saturation)
        windows = train_set.n_windows_per_trajectory
        indices = []
        for t in trajectories:
            first = sum(windows[:t])
            indices += range(first, first + windows[t])
        train_set = Subset(train_set, indices)
    valid_set = WellDataset(path=train.DATASET, well_split_name="valid")
    # evenly spaced (m_t, m_{t+1}) pairs, so validation covers every trajectory and field
    valid_set = Subset(
        valid_set, range(0, len(valid_set), len(valid_set) // N_VALID_SAMPLES)
    )
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=3,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=train.BATCH_SIZE,
        num_workers=3,
        pin_memory=True,
    )

    n, dx = train.data_grid(train.DATASET)
    n_coarse = [k // train.COARSEN for k in n]
    dx_coarse = [dx * train.COARSEN] * 3
    model = make_model(dx_coarse)
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

    # step 0: plain coarse physics, the baseline
    log = [
        {"step": 0}
        | {
            name: value.item()
            for name, value in evaluate(
                model, demag_field, exchange_field, valid_loader
            ).items()
        }
    ]
    print(log[0], flush=True)
    window = []
    for step, batch in enumerate(train.forever(train_loader), start=1):
        window.append(
            train_step(model, optimizer, scheduler, demag_field, exchange_field, batch)
        )

        if step % LOG_EVERY == 0:
            entry = {"step": step, "lr": scheduler.get_last_lr()[0]}
            entry |= {
                name: torch.stack([w[name] for w in window]).mean().item()
                for name in window[0]
            }
            window = []
            if step % VALID_EVERY == 0:
                valid = evaluate(model, demag_field, exchange_field, valid_loader)
                entry |= {name: value.item() for name, value in valid.items()}
                torch.save(model.state_dict(), RUN_DIR / "model.pt")
            log.append(entry)
            print(entry, flush=True)
            (RUN_DIR / "log.json").write_text(json.dumps(log, indent=1))

        if step == N_STEPS:
            break


if __name__ == "__main__":
    main()
