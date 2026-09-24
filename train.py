import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from the_well.benchmark.metrics import MSE, VRMSE
from the_well.data import WellDataset
from tqdm import tqdm

from neuralop.models import CODANO
from normalized_fno import NormalizedFNO
from utils import (
    H_RANGE,
    mse_loss,
    one_step_preds,
    prepare_batch_h_field,
    rollout,
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


# model = NormalizedFNO(
#     n_modes=(16, 16),
#     in_channels=IN_FRAMES * F,
#     out_channels=1 * F,
#     hidden_channels=64,
#     n_layers=2,
#     norm="ada_in",
#     ada_in_features=2,  # H = (Hx, Hy), Hz is always 0
#     factorization="Tucker",
#     rank=0.1,
# ).to(device)


model = CODANO(
    n_layers=2,
    n_modes=[[16, 16], [16, 16]],
    static_channel_dim=2,
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

train_batches = len(train_loader)
val_batches = len(val_loader)

for epoch in range(EPOCHS):
    train_loss = 0
    model.train()
    for batch in tqdm(train_loader, mininterval=10):
        x, y, h = prepare_batch_h_field(batch, device)

        fx = model(x, meta=h)
        loss = mse_loss(y, fx)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
    stats["train_history"].append(train_loss / train_batches)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, mininterval=10):
            x, y, h = prepare_batch_h_field(batch, device)

            fx = model(x, meta=h)
            loss = mse_loss(y, fx)

            val_loss += loss.item()
    stats["val_history"].append(val_loss / val_batches)

    print(f"\n=== epoch {epoch + 1} / {EPOCHS}. loss={val_loss / val_batches:.4f} ===")


model.eval()
torch.save(model.state_dict(), f"{results_path}/model.pt")

### validation
metrics = {"mse": MSE(), "vrmse": VRMSE()}


loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
)
pred, truth = one_step_preds(model, loader, device)
metrics = {
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
