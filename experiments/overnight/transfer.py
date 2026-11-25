"""Do the corrections trained on the small magnet (100 x 50 x 20 cells of 5 nm) transfer to a 4x larger
one? Whole test loops of datasets/shield_medium_5nm (200 x 100 x 20 cells, same field axes) at 25 and
50 nm: coarse physics, the paper's renormalised exchange (paper sigma and the sigma calibrated on the
small magnet by benchmark.py) and the learned models below, none of them trained on the medium magnet.

    python experiments/overnight/transfer.py
"""

import json
from pathlib import Path

import torch

import benchmark
import rollout
import train
import train_torque
from datagen import A, DX, Ms
from model import DemagField, ExchangeField

MODELS = {  # COARSEN -> run directories (all energy models)
    5: ["results/torque_energy_25nm", "results/overnight/lc_25nm_24traj"],
    10: ["results/torque_energy_stable_50nm", "results/overnight/ladder_50nm"],
}

train.DATASET = "datasets/shield_medium_5nm"
train_torque.ENERGY = True
report = {}
for coarsen, run_dirs in MODELS.items():
    train.COARSEN = coarsen
    dx_coarse = [d * coarsen for d in DX]
    m_data, fields, angles = rollout.load_trajectories("test")
    n_coarse = list(m_data.shape[2:5])
    demag_field = DemagField(n=n_coarse, dx=dx_coarse, Ms=Ms, device=train.DEVICE)

    def run(model, A_exchange):
        exchange = ExchangeField(dx=dx_coarse, A=A_exchange, Ms=Ms)
        states, converged = rollout.rollout(model, demag_field, exchange, m_data[:, 0], fields)
        return benchmark.loop_metrics(states, m_data, angles) | {"converged": converged.float().mean().item()}

    sigma_best = json.loads(Path(f"results/benchmark_{coarsen * 5}nm.json").read_text())["sigma_best"]
    results = {
        "coarse physics": run(rollout.NoCorrection(), A),
        "renormalised exchange (paper sigma)": run(rollout.NoCorrection(), rollout.SIGMA * dx_coarse[0] / 2),
        f"renormalised exchange (sigma {sigma_best * 1e3:.2g} mJ/m^2, calibrated on the small magnet)": run(
            rollout.NoCorrection(), sigma_best * dx_coarse[0] / 2
        ),
    }
    for run_dir in run_dirs:
        if not Path(run_dir, "model.pt").exists():
            continue
        model = train_torque.make_model(dx_coarse)
        model.load_state_dict(torch.load(Path(run_dir, "model.pt"), map_location=train.DEVICE))
        model.eval()
        results[run_dir] = run(model, A)
    report[f"{coarsen * 5} nm"] = results
    for name, result in results.items():
        print(f"{coarsen * 5} nm, {name}: {result}", flush=True)
out = Path("results/overnight")
out.mkdir(parents=True, exist_ok=True)
(out / "transfer_medium.json").write_text(json.dumps(report, indent=1))
