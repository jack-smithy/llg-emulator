"""Whole hysteresis loops on the coarse grid, each field step relaxed from the model's own previous
state, against the block-averaged 5 nm data of the test split. One-step validation (train.evaluate)
starts every step from the true m_t; here errors can accumulate and switching can happen at the
wrong field. Three coarse models:

- coarse physics: demag + exchange (A) + applied field;
- renormalised exchange: the paper's coarse domain solver (Shield_Optimization_DS.pdf, section 10),
  exchange replaced by the wall field H_wall = sigma / (mu0 Ms dx) sum_nn (m_j - m_i) (its eq. 7),
  i.e. exchange with A_eff = sigma dx / 2, at the paper's sigma;
- learned correction: coarse physics + h_theta.
"""

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import train
import train_torque
from datagen import A, DX, Ms
from model import DemagField, ExchangeField

RUN_DIR = Path("results/torque_25nm")  # holds model.pt; the report and plot are written here
# the model type comes from train_torque.ENERGY
SPLIT = "test"
# [J/m^2] the paper's wall energy density; its eq. 6, 4 sqrt(A Keff), would give 9.1e-3
SIGMA = 4.6e-3


class NoCorrection(torch.nn.Module):
    """Plain coarse physics: h_theta = 0."""

    def field(self, m):
        return torch.zeros_like(m)


def load_trajectories(split: str) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Coarse m (B, T, X, Y, Z, 3), applied fields (B, T, 3) [A/m] and field axes (B,) [rad]."""
    ms, fields, angles = [], [], []
    for path in sorted(Path(train.DATASET, "data", split).glob("*.hdf5")):
        with h5py.File(path) as f:
            m = torch.as_tensor(f["t1_fields/m"][0], device=train.DEVICE)  # (T, 100, 50, 20, 3)
            b = np.stack([f["scalars/Bx"][0], f["scalars/By"][0], np.zeros_like(f["scalars/Bx"][0])], axis=-1)
            ms.append(train.coarsen(m))
            fields.append(torch.as_tensor(b / train.mu_0, device=train.DEVICE, dtype=torch.float32))
            angles.append(float(f["scalars/field_angle"][0]))
    return torch.stack(ms), torch.stack(fields), np.array(angles)


def rollout(model, demag_field, exchange_field, m0: torch.Tensor, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Relax step by step from m0 (B, X, Y, Z, 3) through fields (B, T, 3): states (B, T, ...), converged (B, T)."""
    states, converged = [m0], [torch.ones(m0.shape[0], dtype=torch.bool, device=m0.device)]
    m = m0
    for t in range(1, fields.shape[1]):
        m, ok = train.relax(model, demag_field, exchange_field, m, fields[:, t])
        states.append(m)
        converged.append(ok)
    return torch.stack(states, dim=1), torch.stack(converged, dim=1)


def main():
    m_data, fields, angles = load_trajectories(SPLIT)  # (B, T, X, Y, Z, 3)
    n_coarse = list(m_data.shape[2:5])  # the magnet's size comes from the data
    dx_coarse = [d * train.COARSEN for d in DX]
    demag_field = DemagField(n=n_coarse, dx=dx_coarse, Ms=Ms, device=train.DEVICE)
    exchange_field = ExchangeField(dx=dx_coarse, A=A, Ms=Ms)
    learned = train_torque.make_model(dx_coarse)
    learned.load_state_dict(torch.load(RUN_DIR / "model.pt", map_location=train.DEVICE))
    learned.eval()

    axes = torch.as_tensor(np.stack([np.cos(angles), np.sin(angles), 0 * angles], -1), device=train.DEVICE, dtype=torch.float32)
    b_axis = (fields * axes[:, None]).sum(-1).cpu().numpy() * train.mu_0 * 1e3  # (B, T) [mT]

    report, loops = {}, {"data": (m_data.mean(dim=(2, 3, 4)) * axes[:, None]).sum(-1).cpu().numpy()}
    def renormalised(sigma):
        return ExchangeField(dx=dx_coarse, A=sigma * dx_coarse[0] / 2, Ms=Ms)  # cubic cells: one A_eff

    # (name, model, exchange, plot colour, line style)
    models = [
        ("coarse physics", NoCorrection(), exchange_field, "#d95f02", "--"),
        (f"renormalised exchange, paper σ {SIGMA * 1e3:.2g} mJ/m²", NoCorrection(), renormalised(SIGMA), "#7570b3", "-."),
    ]
    benchmark = Path("results", f"benchmark_{dx_coarse[0] * 1e9:.0f}nm.json")
    if benchmark.exists():  # sigma calibrated on the valid loops by benchmark.py
        sigma = json.loads(benchmark.read_text())["sigma_best"]
        name = f"renormalised exchange, calibrated σ {sigma * 1e3:.2g} mJ/m²"
        models.append((name, NoCorrection(), renormalised(sigma), "#e7298a", ":"))
    models.append(("learned correction", learned, exchange_field, "#1b9e77", "-"))
    for name, model, exchange, _, _ in models:
        states, converged = rollout(model, demag_field, exchange, m_data[:, 0], fields)
        mse = ((states - m_data) ** 2).sum(-1).mean(dim=(2, 3, 4))  # (B, T)
        loops[name] = (states.mean(dim=(2, 3, 4)) * axes[:, None]).sum(-1).cpu().numpy()
        report[name] = {
            "mse_mean": mse.mean().item(),
            "mse_per_trajectory": mse.mean(dim=1).tolist(),
            "loop_error_mean": float(np.abs(loops[name] - loops["data"]).mean()),
            "converged": converged.float().mean().item(),
        }
        print(name, json.dumps(report[name]), flush=True)
    (RUN_DIR / f"rollout_{SPLIT}.json").write_text(json.dumps(report, indent=1))

    fig, axs = plt.subplots(1, len(angles), figsize=(4 * len(angles), 3.6), sharey=True, layout="constrained")
    styles = [("data", "#222222", "-")] + [(name, color, style) for name, _, _, color, style in models]
    for i, ax in enumerate(axs):
        for name, color, linestyle in styles:
            ax.plot(b_axis[i], loops[name][i], color=color, linestyle=linestyle, linewidth=1.4, label=name)
        ax.set_title(f"field axis {np.degrees(angles[i]):.0f}°")
        ax.set_xlabel("μ0 H along axis (mT)")
    axs[0].set_ylabel("<m> along axis")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.savefig(RUN_DIR / f"rollout_{SPLIT}.png", dpi=150)


if __name__ == "__main__":
    main()
