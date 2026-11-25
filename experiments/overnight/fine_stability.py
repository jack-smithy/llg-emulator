"""Are the stored 5 nm states minima of the 5 nm energy, or did Barzilai-Borwein stop on saddles?
For every 10th field step of the 4 test trajectories of datasets/shield, the softest curvature of the
fine energy (demag + exchange + Zeeman) on the sphere, by train_torque.softest_curvature (Lanczos).
A zero-initialised WallEnergy stands in for "no learned term" (its field is 0 but has a graph).

    python experiments/overnight/fine_stability.py [DATASET]   (default datasets/shield)
"""

import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

import train
import train_torque
from datagen import A, DX, N, Ms
from model import DemagField, ExchangeField, WallEnergy

train_torque.N_LANCZOS_STEPS = 60  # the fine tangent space has 2 x 100k dimensions
STEP_STRIDE = 10
BATCH = 8
DATASET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(train.DATASET)

demag_field = DemagField(n=N, dx=DX, Ms=Ms, device=train.DEVICE)
exchange_field = ExchangeField(dx=DX, A=A, Ms=Ms)
model = WallEnergy(Kd=train.KD, V=math.prod(DX), Ms=Ms).to(train.DEVICE)  # E_theta = 0

results = []
for path in sorted(Path(DATASET, "data", "test").glob("*.hdf5")):
    with h5py.File(path) as f:
        m_all = f["t1_fields/m"][0]  # (T, X, Y, Z, 3)
        b = np.stack([f["scalars/Bx"][0], f["scalars/By"][0], np.zeros_like(f["scalars/Bx"][0])], -1) / train.mu_0
    steps = list(range(0, m_all.shape[0], STEP_STRIDE))
    for start in range(0, len(steps), BATCH):
        chunk = steps[start : start + BATCH]
        m = torch.nn.functional.normalize(torch.as_tensor(m_all[chunk], device=train.DEVICE), dim=-1)
        h_ext = torch.as_tensor(b[chunk], dtype=torch.float32, device=train.DEVICE)
        with torch.no_grad():
            kappa = train_torque.softest_curvature(model, demag_field, exchange_field, m, h_ext)
        for step, value in zip(chunk, kappa.tolist()):
            results.append({"file": path.name, "step": step, "mu0H_mT": float(np.linalg.norm(b[step]) * train.mu_0 * 1e3 * np.sign(b[step] @ b[0])), "kappa": value})
    print(path.name, [round(r["kappa"]) for r in results if r["file"] == path.name], flush=True)

kappas = np.array([r["kappa"] for r in results])
summary = {"states": len(kappas), "unstable_fraction": float((kappas < 0).mean()),
           "kappa_quantiles_5_50_95": np.quantile(kappas, [0.05, 0.5, 0.95]).tolist()}
print(json.dumps(summary), flush=True)
out = Path("results/overnight")
out.mkdir(parents=True, exist_ok=True)
(out / f"fine_stability_{DATASET.name}.json").write_text(json.dumps({"summary": summary, "states": results}, indent=1))
