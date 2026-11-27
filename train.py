"""Train the learned wall energy so that the stored truth states are stable equilibria of the coarse model.

Every stored state is a relaxed state of the fine physics at its applied field. On the coarse grid,
coarse demag + exchange + applied field leave a torque on the block-averaged state; the learned energy
E_theta (model.py) must cancel it, and every state must be a minimum, not a saddle:

    L = mean |P h_eff(m)|^2 / Ms^2  +  STABILITY_WEIGHT * mean relu(-kappa / Ms)^2,

kappa the softest curvature of the coarse energy at m (physics.softest_curvature), both averaged over
the magnet's cells. Batches mix geometries: every state is padded with air to the largest grid
(data.StoredStates), which changes nothing in the physics or the learned energy. No relaxation is
needed to train. Every --select-every steps the model rolls out the valid loops of every geometry, and
the checkpoint with the lowest loop error becomes model.pt: torque and stability only constrain the
data states, and long runs can fall into spurious minima elsewhere.

    python train.py RUN --geometries G1 G3 --coarsen 2 --steps 10000 --seed 0 [--resume]

Writes results/RUN/{args.json, log.json, model.pt, checkpoint.pt, rollout_valid.json,
rollout_test.json}. With --resume a run continues from checkpoint.pt, so a long run can be a chain of
short jobs (scripts/train.slrm).
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import rollout
from data import GEOMETRIES, StoredStates, coarse_grid
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

    start_time = time.time()
    run_dir = Path("results", args.run)
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "rollout_test.json").exists():
        print(f"{run_dir} already exists", flush=True)
        return

    checkpoint_path = run_dir / "checkpoint.pt"
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
    log, best = [], {"loop_error": math.inf}

    window = []

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
            if step == args.steps:
                model.eval()
                valid, _ = rollout.evaluate(model, GEOMETRIES, args.coarsen, "valid")
                model.train()
                entry |= {
                    key: value
                    for key, value in valid.items()
                    if key.startswith("loop_error")
                }
                if valid["loop_error"] < best["loop_error"]:
                    best = {"loop_error": valid["loop_error"], "step": step}
                    torch.save(model.state_dict(), run_dir / "model.pt")
                    (run_dir / "rollout_valid.json").write_text(json.dumps(valid))
            # every LOG_EVERY steps (a small file), so a job cut off by its time limit loses little
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "best": best,
            }
            torch.save(checkpoint, checkpoint_path)
            log.append(entry)
            print(entry, flush=True)
            (run_dir / "log.json").write_text(json.dumps(log, indent=1))

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=DEVICE))
    model.eval()
    test, _ = rollout.evaluate(model, GEOMETRIES, args.coarsen, "test")
    test["selected_step"] = best["step"]
    (run_dir / "rollout_test.json").write_text(json.dumps(test))
    print(
        f"selected step {best['step']}: valid {best['loop_error']:.4f}, test {test['loop_error']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
