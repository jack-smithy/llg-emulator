import torch
from einops import rearrange
from neuralop.models import FNO
from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from tqdm import tqdm

from utils import mse_loss, prepare_batch

device = "cuda"
dataset_path = "datasets/permalloy_thin_film_switching"

IN_CONTEXT_N = 4
LOG_EVERY = 10
N_STEPS = 100
BATCH_SIZE = 16
NUM_WORKERS = -1

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

model = FNO(
    n_modes=(16, 16),
    in_channels=IN_CONTEXT_N * F,
    out_channels=1 * F,
    hidden_channels=128,
    n_layers=5,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

val_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

train_losses = []
val_losses = []

with tqdm(range(5)) as bar:
    for epoch in bar:
        epoch_train_history = []
        epoch_val_history = []

        for batch in train_loader:
            x, y = prepare_batch(batch, device=device)

            fx = model(x)

            loss = mse_loss(y, fx)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_train_history.append(loss.item())

        for batch in val_loader:
            x, y = prepare_batch(batch, device=device)

            fx = model(x)

            loss = mse_loss(y, fx)
            epoch_val_history.append(loss.item())

            bar.set_description(f"loss={loss.item():.4f}")
