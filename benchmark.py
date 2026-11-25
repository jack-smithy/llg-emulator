"""Whole-loop benchmark at one coarse cell size (train.COARSEN), on the test trajectories:

- coarse physics (demag + exchange A + applied field);
- the paper's renormalised exchange (rollout.py): A_eff = sigma dx / 2, at the paper's sigma and at
  the sigma that fits the validation loops best (sigma is the paper's single calibration parameter);
- every learned correction in RUNS.

All are rolled out with rollout.rollout. Writes results/benchmark_<cell size>nm.json.
"""

import json
from pathlib import Path

import numpy as np
import torch

import rollout
import train
import train_torque
from datagen import A, DX, Ms
from model import DemagField, ExchangeField

SIGMA_FACTORS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4]  # times rollout.SIGMA, for the calibration on the valid loops
RUNS = {  # COARSEN -> (run directory, ENERGY)
    5: [("torque_25nm", False), ("torque_energy_25nm", True)],
    10: [("torque_50nm", False), ("torque_energy_50nm", True), ("torque_energy_stable_50nm", True)],
}


def loop_metrics(states: torch.Tensor, m_data: torch.Tensor, angles: np.ndarray) -> dict[str, float]:
    """Loop error mean |<m.axis> - data| and pointwise MSE over all trajectories and steps."""
    axes = torch.as_tensor(np.stack([np.cos(angles), np.sin(angles), 0 * angles], -1), device=states.device, dtype=states.dtype)
    loop = (states.mean(dim=(2, 3, 4)) * axes[:, None]).sum(-1)
    loop_data = (m_data.mean(dim=(2, 3, 4)) * axes[:, None]).sum(-1)
    mse = ((states - m_data) ** 2).sum(-1).mean()
    return {"loop_error": (loop - loop_data).abs().mean().item(), "mse": mse.item()}


def main():
    dx_coarse = [d * train.COARSEN for d in DX]
    trajectories = {split: rollout.load_trajectories(split) for split in ("valid", "test")}
    n_coarse = list(trajectories["test"][0].shape[2:5])  # the magnet's size comes from the data
    demag_field = DemagField(n=n_coarse, dx=dx_coarse, Ms=Ms, device=train.DEVICE)
    exchange_field = ExchangeField(dx=dx_coarse, A=A, Ms=Ms)

    def run(split, model, exchange):
        m_data, fields, angles = trajectories[split]
        states, converged = rollout.rollout(model, demag_field, exchange, m_data[:, 0], fields)
        return loop_metrics(states, m_data, angles) | {"converged": converged.float().mean().item()}

    def renormalised(sigma):
        return ExchangeField(dx=dx_coarse, A=sigma * dx_coarse[0] / 2, Ms=Ms)  # cubic cells

    report = {"cell_size_nm": dx_coarse[0] * 1e9, "sigma_paper": rollout.SIGMA}
    report["sigma_calibration_valid"] = {}
    for factor in SIGMA_FACTORS:
        result = run("valid", rollout.NoCorrection(), renormalised(factor * rollout.SIGMA))
        report["sigma_calibration_valid"][factor * rollout.SIGMA] = result
        print(f"valid, sigma {factor * rollout.SIGMA * 1e3:.2f} mJ/m^2: {result}", flush=True)
    best_sigma = min(report["sigma_calibration_valid"], key=lambda s: report["sigma_calibration_valid"][s]["loop_error"])
    report["sigma_best"] = best_sigma

    report["test"] = {
        "coarse physics": run("test", rollout.NoCorrection(), exchange_field),
        "renormalised exchange (paper sigma)": run("test", rollout.NoCorrection(), renormalised(rollout.SIGMA)),
        f"renormalised exchange (best sigma {best_sigma * 1e3:.2f} mJ/m^2)": run("test", rollout.NoCorrection(), renormalised(best_sigma)),
    }
    for run_dir, energy in RUNS[train.COARSEN]:
        train_torque.ENERGY = energy
        model = train_torque.make_model(dx_coarse)
        model.load_state_dict(torch.load(Path("results", run_dir, "model.pt"), map_location=train.DEVICE))
        model.eval()
        report["test"][run_dir] = run("test", model, exchange_field)
    for name, result in report["test"].items():
        print(f"test, {name}: {result}", flush=True)
    Path("results", f"benchmark_{dx_coarse[0] * 1e9:.0f}nm.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
