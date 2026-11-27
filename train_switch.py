"""train_torque.py (energy + stability) plus a switching loss against pinned states.

Where the data switches between stored steps (mean |m_{t+1} - m_t|^2 above SWITCH_THRESHOLD), the
current model relaxes from the data state m_t at the next field (train.relax, no gradients). If it
stays pinned (closer to m_t than to m_{t+1}), the relaxed state m* is a minimum of the coarse energy
that the true system does not have, so its softest curvature is pushed below zero:

    L_switch = mean over pinned samples of relu(kappa(m*) / Ms + SWITCH_MARGIN)^2

(kappa differentiable in theta at fixed m*, as train_torque.softest_curvature). The model then has to
lose that minimum, i.e. switch at the field where the data switches.

    python train_switch.py RUN_NAME STABILITY_WEIGHT SWITCH_WEIGHT SEED [plain|sym|ctx|res [SWITCH_MARGIN [SOFTNESS_WEIGHT [SOFTNESS [FOLD_WEIGHT [INIT_RUN|- [OFF_DATA_WEIGHT [select]]]]]]]]

(sym: model.SymmetricWallEnergy; ctx: model.ContextWallEnergy, which also sees h_demag and h_ext)

writes results/switch/RUN_NAME/{model.pt, log.json, loops.json}; loops.json holds the rollouts of the
valid and test loops (rollout.rollout_loops).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from the_well.data import WellDataset
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

import rollout
import train
import train_torque
from datagen import A, Ms
from model import ContextWallEnergy, DemagField, ExchangeField, SymmetricWallEnergy, WallEnergyResidual

COARSEN = 10  # 50 nm, as results/torque_energy_stable_50nm
SWITCH_THRESHOLD = 0.02  # mean |m_{t+1} - m_t|^2 per coarse cell above which a step counts as a switch
SWITCH_MARGIN = 5e-3  # push kappa / Ms of pinned states below -SWITCH_MARGIN (typical kappa / Ms of data states: 0.016)
SWITCH_BATCH = 8
SOFTNESS = 4e-3  # kappa / Ms of the last state before a switch: at most about one field step's change
OFF_DATA_NOISE = 0.3  # std of the noise per component of m for off_data_loss
SELECT_EVERY = 2500  # steps between the checkpoints compared by valid rollout


def switching_indices(train_set) -> list[int]:
    """Indices of the training windows whose step is a switch."""
    loader = DataLoader(train_set, batch_size=64, num_workers=3)
    jumps = []
    with torch.no_grad():
        for batch in loader:
            m_start, m_target, _ = train.unpack(batch)
            jumps.append(((m_target - m_start) ** 2).sum(-1).mean(dim=(1, 2, 3)).cpu())
    jumps = torch.cat(jumps)
    print(f"jump quantiles 50/90/99 %: {jumps.quantile(torch.tensor([0.5, 0.9, 0.99])).tolist()}", flush=True)
    return torch.nonzero(jumps > SWITCH_THRESHOLD).flatten().tolist()


def switch_loss(model, demag_field, exchange_field, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """L_switch over the pinned samples of a batch of switching steps, and the pinned fraction."""
    m_start, m_target, h_ext = train.unpack(batch)
    with torch.no_grad():
        m_relaxed, _ = train.relax(model, demag_field, exchange_field, m_start, h_ext)
    distance_start = ((m_relaxed - m_start) ** 2).sum(-1).mean(dim=(1, 2, 3))
    distance_target = ((m_relaxed - m_target) ** 2).sum(-1).mean(dim=(1, 2, 3))
    pinned = distance_start < distance_target
    if not pinned.any():
        return m_start.new_zeros(()), pinned.float().mean()
    kappa = train_torque.softest_curvature(model, demag_field, exchange_field, m_relaxed[pinned], h_ext[pinned]) / Ms
    return (torch.relu(kappa + SWITCH_MARGIN) ** 2).mean(), pinned.float().mean()


def input_field(batch) -> torch.Tensor:
    """The applied field at the input step t, h (B, 3) [A/m] (train.unpack gives the one at t+1)."""
    b = batch["input_scalars"][:, 0, :3].to(train.DEVICE)  # (B, 3): Bx, By, Bz [T]
    if b.shape[1] == 2:  # data without Bz
        b = torch.cat([b, torch.zeros_like(b[:, :1])], dim=1)
    return b / train.mu_0


def softness_loss(model, demag_field, exchange_field, batch) -> torch.Tensor:
    """The last data state before a switch, m_t at its own field H_t, must be stable but only just: its
    branch of minima ends within the next field step, so kappa / Ms in [0, SOFTNESS] (one 4 mT step moves
    kappa / Ms by about that much). Penalises both sides: too stiff pins, unstable switches early."""
    m_start, _, _ = train.unpack(batch)
    kappa = train_torque.softest_curvature(model, demag_field, exchange_field, m_start, input_field(batch)) / Ms
    return (torch.relu(-kappa) ** 2 + torch.relu(kappa - SOFTNESS) ** 2).mean()


def fold_loss(model, demag_field, exchange_field, batches) -> tuple[torch.Tensor, torch.Tensor]:
    """Switch timing at the model's own minima. Data states are not exact minima of the coarse model, and
    its switching is decided at its own minimum next to each data state: m_t* = relax(m_t) at H_t (no
    gradients). Where the data switches between t and t+1, the branch through m_t* must end within the
    step: kappa(m_t*) / Ms in [0, SOFTNESS]; where it does not, one step must not end it: kappa / Ms >=
    SOFTNESS. Returns the loss and the fraction of steps outside their band."""
    m_start = torch.cat([train.unpack(batch)[0] for batch in batches])
    m_target = torch.cat([train.unpack(batch)[1] for batch in batches])
    h_start = torch.cat([input_field(batch) for batch in batches])
    switch = ((m_target - m_start) ** 2).sum(-1).mean(dim=(1, 2, 3)) > SWITCH_THRESHOLD
    with torch.no_grad():
        m_own, _ = train.relax(model, demag_field, exchange_field, m_start, h_start)
    kappa = train_torque.softest_curvature(model, demag_field, exchange_field, m_own, h_start) / Ms
    violation = torch.where(switch, torch.relu(-kappa) + torch.relu(kappa - SOFTNESS), torch.relu(SOFTNESS - kappa))
    return (violation**2).mean(), (violation > 0).float().mean()


def off_data_loss(model, m: torch.Tensor) -> torch.Tensor:
    """Keep the correction small where the data says nothing: mean |h_theta|^2 / Ms^2 at the data states
    tilted by Gaussian noise (OFF_DATA_NOISE per component, renormalised). The torque and stability
    losses only see data states; without this the energy can build deep minima away from them (a
    30000-step run fell into a demagnetised state that no field up to 200 mT could leave)."""
    m_off = torch.nn.functional.normalize(m + OFF_DATA_NOISE * torch.randn_like(m), dim=-1)
    return (model.field(m_off) ** 2).sum(-1).mean() / Ms**2


def main():
    run_name, stability_weight, switch_weight, seed = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    kind = sys.argv[5] if len(sys.argv) > 5 else "plain"  # plain, sym (SymmetricWallEnergy), ctx (ContextWallEnergy) or res (WallEnergyResidual)
    global SWITCH_MARGIN
    if len(sys.argv) > 6:  # one 4 mT step changes kappa / Ms by only ~0.004, so a margin above that fights stability
        SWITCH_MARGIN = float(sys.argv[6])
    softness_weight = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0  # softness_loss
    global SOFTNESS
    if len(sys.argv) > 8:
        SOFTNESS = float(sys.argv[8])
    fold_weight = float(sys.argv[9]) if len(sys.argv) > 9 else 0.0  # fold_loss
    # fine-tune: start from results/switch/INIT_RUN/model.pt, 1000 steps at a tenth of the learning rate
    init_run = sys.argv[10] if len(sys.argv) > 10 and sys.argv[10] != "-" else None  # "-": no fine-tune
    if init_run is not None:
        train_torque.N_STEPS, train_torque.WARMUP_STEPS = 1000, 50
        train_torque.LEARNING_RATE = train_torque.LEARNING_RATE / 10
    off_data_weight = float(sys.argv[11]) if len(sys.argv) > 11 else 0.0  # off_data_loss
    select = len(sys.argv) > 12 and sys.argv[12] == "select"  # keep the checkpoint with the best valid rollout
    run_dir = Path("results/switch") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    train.COARSEN = COARSEN
    train_torque.ENERGY = True

    train_set = WellDataset(path=train.DATASET, well_split_name="train")
    switch_set = Subset(train_set, switching_indices(train_set))
    print(f"{len(switch_set)} switching steps of {len(train_set)}", flush=True)
    valid_set = WellDataset(path=train.DATASET, well_split_name="valid")
    valid_set = Subset(valid_set, range(0, len(valid_set), len(valid_set) // train_torque.N_VALID_SAMPLES))
    train_loader = DataLoader(train_set, batch_size=train_torque.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=3, pin_memory=True)
    switch_loader = DataLoader(switch_set, batch_size=SWITCH_BATCH, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=train.BATCH_SIZE, num_workers=3, pin_memory=True)

    n, dx = train.data_grid(train.DATASET)
    n_coarse = [k // COARSEN for k in n]
    dx_coarse = [dx * COARSEN] * 3
    demag_field = DemagField(n=n_coarse, dx=dx_coarse, Ms=Ms, device=train.DEVICE)
    exchange_field = ExchangeField(dx=dx_coarse, A=A, Ms=Ms)
    model = train_torque.make_model(dx_coarse)
    if kind == "sym":
        model = SymmetricWallEnergy(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms).to(train.DEVICE)
    if kind == "res":
        model = WallEnergyResidual(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms).to(train.DEVICE)
    if kind == "ctx":
        model = ContextWallEnergy(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms, demag_field=demag_field).to(train.DEVICE)
    if init_run is not None:
        model.load_state_dict(torch.load(Path("results/switch", init_run, "model.pt"), map_location=train.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_torque.LEARNING_RATE)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.01, total_iters=train_torque.WARMUP_STEPS),
            CosineAnnealingLR(optimizer, T_max=train_torque.N_STEPS - train_torque.WARMUP_STEPS),
        ],
        milestones=[train_torque.WARMUP_STEPS],
    )

    log, window = [], []
    switch_batches = train.forever(switch_loader)
    for step, batch in enumerate(train.forever(train_loader), start=1):
        _, m, h_ext = train.unpack(batch)
        torque = train_torque.torque_loss(model, demag_field, exchange_field, m, h_ext).mean()
        kappa = train_torque.softest_curvature(model, demag_field, exchange_field, m, h_ext) / Ms
        instability = (torch.relu(-kappa) ** 2).mean()
        loss = torque + stability_weight * instability
        entry = {"torque_loss": torque.item(), "instability": instability.item(), "unstable_fraction": (kappa < 0).float().mean().item()}
        if switch_weight > 0:
            switching, pinned = switch_loss(model, demag_field, exchange_field, next(switch_batches))
            loss = loss + switch_weight * switching
            entry |= {"switch_loss": switching.item(), "pinned_fraction": pinned.item()}
        if softness_weight > 0:
            softness = softness_loss(model, demag_field, exchange_field, next(switch_batches))
            loss = loss + softness_weight * softness
            entry |= {"softness_loss": softness.item()}
        if fold_weight > 0:  # switching steps and the ordinary ones of this batch
            fold, outside = fold_loss(model, demag_field, exchange_field, [next(switch_batches), batch])
            loss = loss + fold_weight * fold
            entry |= {"fold_loss": fold.item(), "fold_outside_band": outside.item()}
        if off_data_weight > 0:
            off_data = off_data_loss(model, m)
            loss = loss + off_data_weight * off_data
            entry |= {"off_data_loss": off_data.item()}

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        window.append(entry)

        if step % train_torque.LOG_EVERY == 0:
            summary = {"step": step} | {key: float(np.mean([w[key] for w in window])) for key in window[0]}
            window = []
            if step % train_torque.VALID_EVERY == 0:
                valid = train_torque.evaluate(model, demag_field, exchange_field, valid_loader)
                summary |= {key: value.item() for key, value in valid.items()}
                torch.save(model.state_dict(), run_dir / "model.pt")
            if select and step % SELECT_EVERY == 0:
                torch.save(model.state_dict(), run_dir / f"model_step{step}.pt")
            log.append(summary)
            print(summary, flush=True)
            (run_dir / "log.json").write_text(json.dumps(log, indent=1))
        if step == train_torque.N_STEPS:
            break

    model.eval()
    report = {}
    if select:  # the checkpoint whose rollout of the valid loops is best becomes model.pt
        valid_trajectories = rollout.load_trajectories("valid")
        scores = {}
        for path in sorted(run_dir.glob("model_step*.pt")):
            model.load_state_dict(torch.load(path, map_location=train.DEVICE))
            metrics, _ = rollout.rollout_loops(model, demag_field, exchange_field, *valid_trajectories)
            scores[path.name] = metrics["loop_error_mean"]
            print(f"{path.name}: valid loop error {scores[path.name]:.4f}", flush=True)
        best = min(scores, key=lambda name: scores[name])
        model.load_state_dict(torch.load(run_dir / best, map_location=train.DEVICE))
        torch.save(model.state_dict(), run_dir / "model.pt")
        report["selected"] = {"checkpoint": best, "valid_loop_errors": scores}
    for split in ("valid", "test"):
        m_data, fields, axes = rollout.load_trajectories(split)
        metrics, loops = rollout.rollout_loops(model, demag_field, exchange_field, m_data, fields, axes)
        report[split] = metrics | {"loops": [loop.tolist() for loop in loops]}
        print(split, {key: metrics[key] for key in ("loop_error_mean", "mse_mean", "converged")}, flush=True)
    (run_dir / "loops.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
