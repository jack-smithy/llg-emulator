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
from datagen import A, Ms
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


def load_trajectories(split: str) -> tuple[list[torch.Tensor], list[torch.Tensor], np.ndarray]:
    """Per trajectory coarse m (T, X, Y, Z, 3) and applied fields (T, 3) [A/m]; field axes (B, 3).

    Lists, not stacked: loops from a random start have fewer steps than loops from saturation.
    """
    ms, fields, axes = [], [], []
    for path in sorted(Path(train.DATASET, "data", split).glob("*.hdf5")):
        with h5py.File(path) as f:
            m = torch.as_tensor(f["t1_fields/m"][0], device=train.DEVICE)  # (T, X, Y, Z, 3)
            bz = f["scalars/Bz"][0] if "scalars/Bz" in f else np.zeros_like(f["scalars/Bx"][0])  # older files: no Bz
            b = np.stack([f["scalars/Bx"][0], f["scalars/By"][0], bz], axis=-1)
            ms.append(train.coarsen(m))
            fields.append(torch.as_tensor(b / train.mu_0, device=train.DEVICE, dtype=torch.float32))
            azimuth = float(f["scalars/field_angle"][0])
            # older files: in-plane axes only
            elevation = float(f["scalars/field_elevation"][0]) if "scalars/field_elevation" in f else 0.0
            axes.append([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    return ms, fields, np.array(axes)


def mean_along(states: torch.Tensor, axis: np.ndarray) -> np.ndarray:
    """<m> along the field axis at every step: states (T, X, Y, Z, 3) -> (T,)."""
    return states.mean(dim=(1, 2, 3)).cpu().numpy() @ axis


def rollout(model, demag_field, exchange_field, m0: torch.Tensor, fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Relax step by step from m0 (B, X, Y, Z, 3) through fields (B, T, 3): states (B, T, ...), converged (B, T)."""
    states, converged = [m0], [torch.ones(m0.shape[0], dtype=torch.bool, device=m0.device)]
    m = m0
    for t in range(1, fields.shape[1]):
        m, ok = train.relax(model, demag_field, exchange_field, m, fields[:, t])
        states.append(m)
        converged.append(ok)
    return torch.stack(states, dim=1), torch.stack(converged, dim=1)


def rollout_loops(model, demag_field, exchange_field, m_data, fields, axes) -> tuple[dict, list[np.ndarray]]:
    """Roll out each loop of load_trajectories on its own: metrics, and <m> along the axis per loop."""
    loops, mse, loop_error, converged = [], [], [], []
    for m_loop, h_loop, axis in zip(m_data, fields, axes):
        states, ok = rollout(model, demag_field, exchange_field, m_loop[None, 0], h_loop[None])
        states = states[0]  # (T, X, Y, Z, 3)
        loops.append(mean_along(states, axis))
        mse.append(((states - m_loop) ** 2).sum(-1).mean().item())
        loop_error.append(float(np.abs(loops[-1] - mean_along(m_loop, axis)).mean()))
        converged.append(ok.float().mean().item())
    metrics = {
        "mse_mean": float(np.mean(mse)),
        "mse_per_trajectory": mse,
        "loop_error_mean": float(np.mean(loop_error)),
        "converged": float(np.mean(converged)),
    }
    return metrics, loops


def main():
    m_data, fields, axes = load_trajectories(SPLIT)
    n_coarse = list(m_data[0].shape[1:4])  # the magnet's size comes from the data
    _, dx = train.data_grid(train.DATASET)
    dx_coarse = [dx * train.COARSEN] * 3
    demag_field = DemagField(n=n_coarse, dx=dx_coarse, Ms=Ms, device=train.DEVICE)
    exchange_field = ExchangeField(dx=dx_coarse, A=A, Ms=Ms)
    learned = train_torque.make_model(dx_coarse)
    learned.load_state_dict(torch.load(RUN_DIR / "model.pt", map_location=train.DEVICE))
    learned.eval()

    report, loops = {}, {"data": [mean_along(m, axis) for m, axis in zip(m_data, axes)]}
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
        report[name], loops[name] = rollout_loops(model, demag_field, exchange, m_data, fields, axes)
        print(name, json.dumps(report[name]), flush=True)
    (RUN_DIR / f"rollout_{SPLIT}.json").write_text(json.dumps(report, indent=1))

    fig, axs = plt.subplots(1, len(axes), figsize=(4 * len(axes), 3.6), sharey=True, layout="constrained")
    styles = [("data", "#222222", "-")] + [(name, color, style) for name, _, _, color, style in models]
    for i, ax in enumerate(axs):
        b_axis = fields[i].cpu().numpy() @ axes[i] * train.mu_0 * 1e3  # (T,) [mT]
        for name, color, linestyle in styles:
            ax.plot(b_axis, loops[name][i], color=color, linestyle=linestyle, linewidth=1.4, label=name)
        azimuth, elevation = np.degrees(np.arctan2(axes[i][1], axes[i][0])), np.degrees(np.arcsin(axes[i][2]))
        ax.set_title(f"field axis {azimuth:.0f}°, {elevation:.0f}° out of plane")
        ax.set_xlabel("μ0 H along axis (mT)")
    axs[0].set_ylabel("<m> along axis")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels), frameon=False, fontsize=8)
    fig.savefig(RUN_DIR / f"rollout_{SPLIT}.png", dpi=150)


if __name__ == "__main__":
    main()
