import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from the_well.benchmark.metrics import MSE, VRMSE
from the_well.data import WellDataset

from model import NormalizedFNO
from utils import (
    H_RANGE,
    mse_loss,
    one_step_preds,
    rollout,
    train_epoch,
    val_epoch,
)

parser = ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--configuration", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--learning-rate", type=float, default=5e-3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--in-frames", type=int, default=4)
args = parser.parse_args()

device = "cuda"
path = f"datasets/{args.dataset}"
results_path = Path("results") / args.dataset / args.configuration / f"seed_{args.seed}"
results_path.mkdir(exist_ok=True, parents=True)

torch.set_default_dtype(torch.float32)
torch.manual_seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)

IN_FRAMES = args.in_frames
OUT_FRAMES = 1
BATCH_SIZE = args.batch_size
NUM_WORKERS = 4
N_FRAMES_ROLLOUT = 100
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate
ROLLOUT_IDX = 7

train_dataset = WellDataset(
    path=path,
    well_split_name="train",
    n_steps_input=IN_FRAMES,
    n_steps_output=OUT_FRAMES,
    use_normalization=False,
)

val_dataset = WellDataset(
    path=path,
    well_split_name="valid",
    n_steps_input=IN_FRAMES,
    n_steps_output=OUT_FRAMES,
    use_normalization=False,
)


F = train_dataset.metadata.n_fields


model = NormalizedFNO(
    n_modes=(16, 16),
    in_channels=IN_FRAMES * F,
    out_channels=1 * F,
    hidden_channels=64,
    n_layers=2,
    norm="ada_in",
    ada_in_features=2,  # H = (Hx, Hy), Hz is always 0
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
    drop_last=True,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
    drop_last=True,
    pin_memory=True,
)


stats: dict = {"train_history": [], "val_history": []}

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, mse_loss, optimizer, device)
    stats["train_history"].append(train_loss)

    val_loss = val_epoch(model, val_loader, mse_loss, device)
    stats["val_history"].append(val_loss)

    print(f"\n=== epoch {epoch + 1} / {EPOCHS}. loss={val_loss:.4f} ===")


model.eval()
torch.save(model.state_dict(), f"{results_path}/model.pt")

### validation
loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
)
pred, truth = one_step_preds(model, loader, device)
stats["metrics"] = {
    "mse": MSE().eval(pred, truth, meta=val_dataset.metadata),
    "vrmse": VRMSE().eval(pred, truth, meta=val_dataset.metadata),
}

### test on sp4
rollout_dataset = WellDataset(
    path=path,
    well_split_name="valid",
    n_steps_input=N_FRAMES_ROLLOUT,
    n_steps_output=OUT_FRAMES,
    use_normalization=False,
)

sample = rollout_dataset[ROLLOUT_IDX]
truth = sample["input_fields"].unsqueeze(0).to(device)
h = (sample["constant_scalars"][3:5]).unsqueeze(0).to(device)
h = (h - H_RANGE[0]) / (H_RANGE[1] - H_RANGE[0])
pred = rollout(model, truth[:, :IN_FRAMES], h, n_steps=truth.shape[1] - IN_FRAMES)
truth = truth[:, IN_FRAMES:]

rollout_data = np.stack([truth[0].cpu().numpy(), pred[0].cpu().numpy()])
np.save(f"{results_path}/rollout.npy", rollout_data)

stats["config"] = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "in_context_n": IN_FRAMES,
    "seed": args.seed,
    "configuration": args.configuration,
}
with open(f"{results_path}/stats.json", "w+") as f:
    json.dump(stats, f)
