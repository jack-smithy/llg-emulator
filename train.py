import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from the_well.benchmark.metrics import MSE, VRMSE
from the_well.data import WellDataset
from tqdm import tqdm

from normalized_fno import NormalizedFNO
from utils import mse_loss, one_step_preds, prepare_batch, rollout

parser = ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--configuration", type=str, required=True)
args = parser.parse_args()

device = "cuda"
dataset_path = "datasets/permalloy_thin_film_switching"
results_path = Path("results/") / args.configuration / f"seed_{args.seed}"
results_path.mkdir(exist_ok=True, parents=True)

torch.manual_seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)

IN_CONTEXT_N = 1
BATCH_SIZE = 16
NUM_WORKERS = 4
N_FRAMES = 100
EPOCHS = 5
LEARNING_RATE = 5e-3

train_dataset = WellDataset(
    path=dataset_path,
    well_split_name="train",
    n_steps_input=IN_CONTEXT_N,
    n_steps_output=1,
    use_normalization=False,
)

val_dataset = WellDataset(
    path=dataset_path,
    well_split_name="valid",
    n_steps_input=IN_CONTEXT_N,
    n_steps_output=1,
    use_normalization=False,
)


F = train_dataset.metadata.n_fields

model = NormalizedFNO(
    n_modes=(16, 16),
    in_channels=IN_CONTEXT_N * F,
    out_channels=1 * F,
    hidden_channels=128,
    n_layers=5,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
)

stats = {"history": []}
with tqdm(range(EPOCHS)) as bar:
    for epoch in bar:
        for batch in train_loader:
            x, y = prepare_batch(batch, device=device)

            fx = model(x)

            loss = mse_loss(y, fx)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            stats["history"].append(loss.item())


model.eval()
torch.save(model.state_dict(), f"{results_path}/model.pt")

### validation
metrics = {"mse": MSE(), "vrmse": VRMSE()}

for split, dataset in [("train", train_dataset), ("val", val_dataset)]:
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        generator=generator,
    )
    pred, truth = one_step_preds(model, loader, device=device)
    # the well's metrics reduce over space, leaving (N, 1, F) to average
    stats[split] = {  # type: ignore
        name: metric(pred, truth, dataset.metadata).mean().item()
        for name, metric in metrics.items()
    }
    print(f"{split} one-step: {stats[split]}")

### test on sp4
test_dataset = WellDataset(
    path=dataset_path,
    well_split_name="test",
    n_steps_input=N_FRAMES,
    n_steps_output=1,
    use_normalization=False,
)

truth = test_dataset[0]["input_fields"].unsqueeze(0).to(device)
pred = rollout(model, truth[:, :IN_CONTEXT_N], n_steps=truth.shape[1] - IN_CONTEXT_N)
truth = truth[:, IN_CONTEXT_N:]

rollout_data = np.stack([truth[0].cpu().numpy(), pred[0].cpu().numpy()])
np.save(f"{results_path}/rollout.npy", rollout_data)


with open(f"{results_path}/stats.json", "w+") as f:
    json.dump(stats, f)
