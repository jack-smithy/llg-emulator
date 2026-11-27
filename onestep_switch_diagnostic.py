"""One-step switching diagnostic, to tell local errors of the learned energy from rollout drift. Data
states are not exact minima of the coarse model, so each is first relaxed at its own field H_t (m_t*,
the model's nearby minimum), then from there at the next field H_{t+1} (m_{t+1}*). Per valid step:

- early: the data does not switch, the model jumps (mean |m_{t+1}* - m_t*|^2 > SWITCH_THRESHOLD);
- pinned: the data switches, the model does not jump;
- switch ok: both switch;

    python onestep_switch_diagnostic.py RUN [RUN ...]
"""

import math
import sys
from pathlib import Path

import torch
from the_well.data import WellDataset
from torch.utils.data import DataLoader

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
loader = DataLoader(WellDataset(path=train.DATASET, well_split_name="valid"), batch_size=100, num_workers=3)


def distance(a, b):
    return ((a - b) ** 2).sum(-1).mean(dim=(1, 2, 3))


for run in sys.argv[1:]:
    if run.startswith("sym"):
        model = SymmetricWallEnergy(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms)
    elif run.startswith("res"):
        model = WallEnergyResidual(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms)
    elif run.startswith("ctx"):
        model = ContextWallEnergy(Kd=train.KD, V=math.prod(dx_coarse), Ms=Ms, demag_field=demag_field)
    else:
        model = train_torque.make_model(dx_coarse)
    model = model.to(train.DEVICE)
    model.load_state_dict(torch.load(Path("results/switch", run, "model.pt"), map_location=train.DEVICE))
    model.eval()
    counts = {"steps": 0, "data switches": 0, "early": 0, "pinned": 0, "switch ok": 0}
    for batch in loader:
        m_start, m_target, h_ext = train.unpack(batch)
        b = batch["input_scalars"][:, 0, :3].to(train.DEVICE)
        if b.shape[1] == 2:
            b = torch.cat([b, torch.zeros_like(b[:, :1])], dim=1)
        with torch.no_grad():
            m_own, _ = train.relax(model, demag_field, exchange_field, m_start, b / train.mu_0)  # m_t* at H_t
            m_next, _ = train.relax(model, demag_field, exchange_field, m_own, h_ext)  # m_{t+1}* at H_{t+1}
        data_switch = distance(m_target, m_start) > train_switch.SWITCH_THRESHOLD
        model_jump = distance(m_next, m_own) > train_switch.SWITCH_THRESHOLD
        closer_to_target = model_jump
        counts["steps"] += len(m_start)
        counts["data switches"] += int(data_switch.sum())
        counts["early"] += int((~data_switch & model_jump).sum())
        counts["pinned"] += int((data_switch & ~closer_to_target).sum())
        counts["switch ok"] += int((data_switch & closer_to_target).sum())
    print(run, counts, flush=True)
