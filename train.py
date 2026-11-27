"""Train the learned wall energy so that the stored truth states are stable equilibria of the coarse model.

Every stored state is a relaxed state of the fine physics at its applied field. On the coarse grid,
coarse demag + exchange + applied field leave a torque on the block-averaged state; the learned energy
E_theta (model.py) must cancel it, and every state must be a minimum, not a saddle:

    L = mean |P h_eff(m)|^2 / Ms^2  +  lambda_stability * mean relu(-kappa / Ms)^2,

kappa the softest curvature of the coarse energy at m (physics.softest_curvature), both averaged over
the magnet's cells. Batches mix the geometries data.GEOMETRIES: every state is padded with air to the
largest grid (data.StoredStates), which changes nothing in the physics or the learned energy. No
relaxation is needed to train. At the end the model rolls out the valid and test loops of every
geometry (rollout.evaluate).

    python train.py RUN --coarsen 2 --steps 10000 --seed 0       (one long job: scripts/train.slrm)

Writes results/RUN/{args.json, log.json, model.pt, rollout_valid.json, rollout_test.json}.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import rollout
from data import GEOMETRIES, StoredStates
from datagen import A, Ms
from model import WallEnergy
from physics import (
    DEVICE,
    KD,
    DemagField,
    ExchangeField,
    effective_field,
    softest_curvature,
    tangent,
)

WARMUP_STEPS = 100
LOG_EVERY = 100


def torque_loss(model, demag_field, exchange_field, m, h_ext, mask):
    """Mean |P h_eff|^2 / Ms^2 over the magnet's cells, per sample (B,)."""
    h = effective_field(model, demag_field, exchange_field, m, h_ext, mask)
    return (tangent(m, h) ** 2).sum(dim=(1, 2, 3, 4)) / mask.sum(dim=(1, 2, 3)) / Ms**2


def train_step(
    model,
    optimizer,
    scheduler,
    demag_field,
    exchange_field,
    batch,
    lambda_stability,
) -> dict[str, torch.Tensor]:
    m, mask, h_ext = (batch[key].to(DEVICE) for key in ("m", "mask", "h_ext"))
    torque = torque_loss(model, demag_field, exchange_field, m, h_ext, mask).mean()
    kappa = softest_curvature(model, demag_field, exchange_field, m, h_ext, mask) / Ms
    instability = (torch.relu(-kappa) ** 2).mean()

    # L = L_torque + lambda * L_stability
    loss = torque + lambda_stability * instability

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    return {
        "torque_loss": torque.detach(),
        "instability": instability.detach(),
        "unstable_fraction": (kappa < 0).float().mean(),
    }


def forever(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", help="run name: writes results/RUN")
    parser.add_argument("--coarsen", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-stability", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    run_dir = Path("results", args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "rollout_test.json").exists():
        print(f"{run_dir} already exists", flush=True)
        return

    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=1))

    torch.manual_seed(args.seed)

    train_set = StoredStates(GEOMETRIES, "train", args.coarsen)

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=3,
        pin_memory=True,
    )

    # the same cell size on every geometry
    dx = train_set.dx  # the coarse cell size: stored 50 nm x coarsen

    demag_field = DemagField(n=train_set.shape, dx=dx, Ms=Ms, device=DEVICE)
    exchange_field = ExchangeField(dx=dx, A=A, Ms=Ms)
    print(
        f"{len(train_set)} states of {GEOMETRIES}, padded to {train_set.shape}",
        flush=True,
    )
    model = WallEnergy(Kd=KD, V=math.prod(dx), Ms=Ms).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_STEPS),
            CosineAnnealingLR(optimizer, T_max=args.steps - WARMUP_STEPS),
        ],
        milestones=[WARMUP_STEPS],
    )
    log, window = [], []

    for step, batch in zip(range(1, args.steps + 1), forever(loader)):
        window.append(
            train_step(
                model,
                optimizer,
                scheduler,
                demag_field,
                exchange_field,
                batch,
                lambda_stability=args.lambda_stability,
            )
        )

        if step % LOG_EVERY == 0:
            entry = {"step": step, "lr": scheduler.get_last_lr()[0]}
            entry |= {
                name: torch.stack([w[name] for w in window]).mean().item()
                for name in window[0]
            }
            window = []
            log.append(entry)
            print(entry, flush=True)
            (run_dir / "log.json").write_text(json.dumps(log, indent=1))

    # saved before the rollouts, so a failing rollout cannot lose the trained model
    torch.save(model.state_dict(), run_dir / "model.pt")
    model.eval()
    loop_errors = {}
    for split in ("valid", "test"):
        report, _ = rollout.evaluate(model, GEOMETRIES, args.coarsen, split)
        (run_dir / f"rollout_{split}.json").write_text(json.dumps(report))
        loop_errors[split] = {
            key: round(value, 4)
            for key, value in report.items()
            if key.startswith("loop_error")
        }
    print(f"loop errors: {loop_errors}", flush=True)


if __name__ == "__main__":
    main()
