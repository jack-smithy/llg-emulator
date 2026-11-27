"""Roll out the train loops of datasets/shield (50 nm) with every model in results/switch, so methods are
compared on 32 loops instead of the 8 valid + test ones. The models never trained on whole loops (only
on single states), so train-loop rollouts still test switching. Writes <run>/loops_train.json, skipping
runs that have it. Runs that hit their time limit before train_switch.py's own rollouts (no loops.json)
get those too, from their last checkpoint (model.pt is saved every VALID_EVERY steps).

    python eval_switch_runs.py [PREFIX ...]   (only runs whose names start with a PREFIX; skips runs
                                               whose sw- training job is still in the queue)
"""

import json
import subprocess
import sys
import math
from pathlib import Path

import torch

import rollout
import train
import train_switch
import train_torque
from datagen import A, Ms
from model import ContextWallEnergy, DemagField, ExchangeField, SymmetricWallEnergy, WallEnergyResidual

train.COARSEN = train_switch.COARSEN
train_torque.ENERGY = True
n, dx = train.data_grid(train.DATASET)
dx_coarse = [dx * train.COARSEN] * 3
demag_field = DemagField(n=[k // train.COARSEN for k in n], dx=dx_coarse, Ms=Ms, device=train.DEVICE)
exchange_field = ExchangeField(dx=dx_coarse, A=A, Ms=Ms)
trajectories = {split: rollout.load_trajectories(split) for split in ("train", "valid", "test")}
# runs whose training job (sw-NAME-sSEED) is still queued or running: their model.pt is not final
jobs = subprocess.run(["squeue", "--me", "-h", "-o", "%j"], capture_output=True, text=True, check=True).stdout.split()
running = {job.removeprefix("sw-").replace("-s", "_s") for job in jobs if job.startswith("sw-")}

only = sys.argv[1:]  # run-name prefixes to evaluate (default: all)
for run_dir in sorted(Path("results/switch").glob("*_s[0-9]")):
    if only and not any(run_dir.name.startswith(prefix) for prefix in only):
        continue
    if (run_dir / "loops_train.json").exists() or not (run_dir / "model.pt").exists() or run_dir.name in running:
        continue
    if run_dir.name.startswith("sym"):
        model = SymmetricWallEnergy(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms)
    elif run_dir.name.startswith("res"):
        model = WallEnergyResidual(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms)
    elif run_dir.name.startswith("ctx"):
        model = ContextWallEnergy(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms, demag_field=demag_field)
    else:
        model = train_torque.make_model(dx_coarse)
    model = model.to(train.DEVICE)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=train.DEVICE))
    model.eval()
    if not (run_dir / "loops.json").exists():  # stopped by its time limit before its own rollouts
        report = {}
        for split in ("valid", "test"):
            metrics, loops = rollout.rollout_loops(model, demag_field, exchange_field, *trajectories[split])
            report[split] = metrics | {"loops": [loop.tolist() for loop in loops]}
        (run_dir / "loops.json").write_text(json.dumps(report))
        print(run_dir.name, "valid", report["valid"]["loop_error_mean"], "test", report["test"]["loop_error_mean"], flush=True)
    metrics, loops = rollout.rollout_loops(model, demag_field, exchange_field, *trajectories["train"])
    (run_dir / "loops_train.json").write_text(json.dumps(metrics | {"loops": [loop.tolist() for loop in loops]}))
    print(run_dir.name, "train", {key: metrics[key] for key in ("loop_error_mean", "mse_mean", "converged")}, flush=True)
