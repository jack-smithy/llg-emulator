"""Benchmark without physics in the model: a plain CNN predicts the next coarse state m_{t+1} from m_t
and the next applied field, trained on one-step MSE. Whole test loops are then rolled out two ways:

- CNN only: each step's prediction is the next step's input;
- CNN + relax: each prediction is relaxed with coarse physics (demag + exchange + applied field,
  train.relax) before it is used, so every state is an equilibrium of coarse physics.

Writes results/next_state_<nm>nm/{model.pt, log.json, rollout_test.json, rollout_test.png}, with
coarse physics alone and the learned energy correction (REFERENCE_RUN) as references.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from the_well.data import WellDataset
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

import rollout
import train
import train_torque
from datagen import A, Ms
from model import DemagField, ExchangeField

COARSEN = 10  # datasets/shield (5 nm) at 50 nm, as results/torque_energy_stable_50nm and benchmark_50nm.json
REFERENCE_RUN = Path("results/torque_energy_stable_50nm")  # learned energy correction (train_torque.ENERGY)
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
N_STEPS = 30000
WARMUP_STEPS = 500
LOG_EVERY = 500
VALID_EVERY = 2500
N_VALID_SAMPLES = 256


class NextState(nn.Module):
    """m_{t+1} = normalise(m_t + CNN(m_t, h_ext / Ms)), with LearnedField's CNN; starts at m_{t+1} = m_t."""

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(6, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.SiLU(),
            nn.Conv3d(32, 3, kernel_size=1),
        )
        nn.init.zeros_(self.cnn[-1].weight)
        nn.init.zeros_(self.cnn[-1].bias)

    def forward(self, m: Tensor, h_ext: Tensor) -> Tensor:
        """m (B, X, Y, Z, 3), h_ext (B, 3) [A/m] -> m_{t+1} (B, X, Y, Z, 3)."""
        h = (h_ext / Ms)[:, None, None, None, :].expand_as(m)
        x = torch.cat([m, h], dim=-1).permute(0, 4, 1, 2, 3)
        dm = self.cnn(x).permute(0, 2, 3, 4, 1)
        return F.normalize(m + dm, dim=-1)


def mse(m: Tensor, m_target: Tensor) -> Tensor:
    """Mean over cells of |m - m_target|^2, per sample: (B,)."""
    return ((m - m_target) ** 2).sum(-1).mean(dim=(1, 2, 3))


def rollout_loop(model, m0: Tensor, fields: Tensor, relax_with=None) -> Tensor:
    """States (T, X, Y, Z, 3) from m0 (X, Y, Z, 3) through fields (T, 3); relax_with = (demag, exchange)
    relaxes every prediction with coarse physics."""
    m, states = m0[None], [m0[None]]
    for t in range(1, len(fields)):
        m = model(m, fields[None, t])
        if relax_with is not None:
            m, _ = train.relax(rollout.NoCorrection(), *relax_with, m, fields[None, t])
        states.append(m)
    return torch.cat(states)


def loop_metrics(states: list[Tensor], m_data: list[Tensor], axes: np.ndarray) -> tuple[dict, list[np.ndarray]]:
    """Metrics as rollout.rollout_loops, and <m> along the axis per loop."""
    loops = [rollout.mean_along(s, a) for s, a in zip(states, axes)]
    loop_error = [np.abs(loop - rollout.mean_along(m, a)).mean() for loop, m, a in zip(loops, m_data, axes)]
    mses = [mse(s, m).mean().item() for s, m in zip(states, m_data)]
    metrics = {"loop_error_mean": float(np.mean(loop_error)), "mse_mean": float(np.mean(mses)), "mse_per_trajectory": mses}
    return metrics, loops


def main():
    torch.manual_seed(0)
    train.COARSEN = COARSEN
    n, dx = train.data_grid(train.DATASET)
    dx_coarse = [dx * train.COARSEN] * 3
    run_dir = Path("results", f"next_state_{dx_coarse[0] * 1e9:.0f}nm")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = WellDataset(path=train.DATASET, well_split_name="train")
    valid_set = WellDataset(path=train.DATASET, well_split_name="valid")
    valid_set = Subset(valid_set, range(0, len(valid_set), max(1, len(valid_set) // N_VALID_SAMPLES)))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=3, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=3, pin_memory=True)

    model = NextState().to(train.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.01, total_iters=WARMUP_STEPS),
            CosineAnnealingLR(optimizer, T_max=N_STEPS - WARMUP_STEPS),
        ],
        milestones=[WARMUP_STEPS],
    )

    def validate():
        with torch.no_grad():
            losses = [mse(model(m_start, h_ext), m_target) for m_start, m_target, h_ext in map(train.unpack, valid_loader)]
            persistence = [mse(m_start, m_target) for m_start, m_target, _ in map(train.unpack, valid_loader)]
        return {"valid_mse": torch.cat(losses).mean().item(), "valid_mse_no_change": torch.cat(persistence).mean().item()}

    log, window = [{"step": 0} | validate()], []
    print(log[0], flush=True)
    for step, batch in enumerate(train.forever(train_loader), start=1):
        m_start, m_target, h_ext = train.unpack(batch)
        loss = mse(model(m_start, h_ext), m_target).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        window.append(loss.item())

        if step % LOG_EVERY == 0:
            entry = {"step": step, "lr": scheduler.get_last_lr()[0], "train_mse": float(np.mean(window))}
            window = []
            if step % VALID_EVERY == 0:
                entry |= validate()
                torch.save(model.state_dict(), run_dir / "model.pt")
            log.append(entry)
            print(entry, flush=True)
            (run_dir / "log.json").write_text(json.dumps(log, indent=1))
        if step == N_STEPS:
            break

    model.eval()
    m_data, fields, axes = rollout.load_trajectories("test")
    demag_field = DemagField(n=list(m_data[0].shape[1:4]), dx=dx_coarse, Ms=Ms, device=train.DEVICE)
    exchange_field = ExchangeField(dx=dx_coarse, A=A, Ms=Ms)
    train_torque.ENERGY = True
    learned = train_torque.make_model(dx_coarse)
    learned.load_state_dict(torch.load(REFERENCE_RUN / "model.pt", map_location=train.DEVICE))
    learned.eval()

    report, loops = {}, {"data": [rollout.mean_along(m, a) for m, a in zip(m_data, axes)]}
    with torch.no_grad():
        states = [rollout_loop(model, m[0], h) for m, h in zip(m_data, fields)]
        report["CNN only"], loops["CNN only"] = loop_metrics(states, m_data, axes)
        states = [rollout_loop(model, m[0], h, (demag_field, exchange_field)) for m, h in zip(m_data, fields)]
        report["CNN + relax"], loops["CNN + relax"] = loop_metrics(states, m_data, axes)
    report["coarse physics"], loops["coarse physics"] = rollout.rollout_loops(
        rollout.NoCorrection(), demag_field, exchange_field, m_data, fields, axes
    )
    report[REFERENCE_RUN.name], loops[REFERENCE_RUN.name] = rollout.rollout_loops(
        learned, demag_field, exchange_field, m_data, fields, axes
    )
    for name, metrics in report.items():
        print(name, {key: metrics[key] for key in ("loop_error_mean", "mse_mean")}, flush=True)
    (run_dir / "rollout_test.json").write_text(json.dumps(report, indent=1))

    styles = {  # name -> (colour, line style)
        "data": ("#222222", "-"),
        "coarse physics": ("#d95f02", "--"),
        "CNN only": ("#7570b3", ":"),
        "CNN + relax": ("#e7298a", "-."),
        REFERENCE_RUN.name: ("#1b9e77", "-"),
    }
    fig, axs = plt.subplots(1, len(axes), figsize=(4 * len(axes), 3.6), sharey=True, layout="constrained")
    for i, ax in enumerate(axs):
        b_axis = fields[i].cpu().numpy() @ axes[i] * train.mu_0 * 1e3  # (T,) [mT]
        for name, (color, linestyle) in styles.items():
            label = name if name == "data" else f"{name} ({report[name]['loop_error_mean']:.3f})"
            ax.plot(b_axis, loops[name][i], color=color, linestyle=linestyle, linewidth=1.4, label=label)
        ax.set_title(f"field axis {np.degrees(np.arctan2(axes[i][1], axes[i][0])):.0f}°")
        ax.set_xlabel("μ0 H along axis (mT)")
    axs[0].set_ylabel("<m> along axis")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(f"test loops at {dx_coarse[0] * 1e9:.0f} nm (mean loop error in brackets)")
    fig.savefig(run_dir / "rollout_test.png", dpi=150)


if __name__ == "__main__":
    main()
