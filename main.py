import json
from argparse import ArgumentParser
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import torch
from the_well.data import WellDataset
from tqdm import tqdm
import pdequinox as pdeqx

from model_config import ModelConfig
from training import build_model, loss_fn, save_model, update_fn
from utils import SCALARS, numpy_collate, prepare_batch

jax.config.update("jax_compilation_cache_dir", ".jax_cache")

parser = ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--configuration", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--learning-rate", type=float, default=5e-3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--hidden-channels", type=int, default=64)
parser.add_argument("--num-blocks", type=int, default=4)
parser.add_argument("--num-modes", type=int, default=32)
args = parser.parse_args()

path = f"datasets/{args.dataset}"
results_path = (
    Path("results")
    / args.dataset
    / args.arch
    / args.configuration
    / f"seed_{args.seed}"
)
results_path.mkdir(exist_ok=True, parents=True)

torch.manual_seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)
key = jr.PRNGKey(args.seed)


IN_FRAMES = 1  # the model is a one-step map m_t -> m_{t+1}
OUT_FRAMES = 1
BATCH_SIZE = args.batch_size
NUM_WORKERS = 4
EPOCHS = args.epochs
LEARNING_RATE = args.learning_rate

stats = {}
stats["config"] = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "in_context_n": IN_FRAMES,
    "hidden_channels": args.hidden_channels,
    "num_blocks": args.num_blocks,
    "seed": args.seed,
    "configuration": args.configuration,
}
with open(f"{results_path}/stats.json", "w+") as f:
    json.dump(stats, f)


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

# `conditioning` indexes constant_scalars by position
assert tuple(train_dataset.metadata.constant_scalar_names) == SCALARS

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
    drop_last=True,
    pin_memory=False,
    collate_fn=numpy_collate,
    persistent_workers=True,
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
    drop_last=True,
    pin_memory=False,
    collate_fn=numpy_collate,
    persistent_workers=True,
)

model_config = ModelConfig(
    hidden_channels=args.hidden_channels,
    num_blocks=args.num_blocks,
    num_modes=args.num_modes,
)

key, model_key = jr.split(key)
model = build_model(model_config, model_key)
print(f"{pdeqx.count_parameters(model)} trainable parameters")

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=LEARNING_RATE, weight_decay=1e-5),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

stats["train_history"] = []
stats["val_history"] = []

for epoch in range(EPOCHS):
    losses = []
    for batch in tqdm(train_loader, mininterval=10):
        batch = prepare_batch(batch)
        model, opt_state, loss = update_fn(model, batch, optimizer, opt_state)
        losses.append(loss)
    stats["train_history"].append(float(jnp.stack(losses).mean()))

    inference_model = eqx.nn.inference_mode(model)
    losses = []
    for batch in tqdm(val_loader, mininterval=10):
        batch = prepare_batch(batch)
        loss = loss_fn(inference_model, *batch)
        losses.append(loss)
    val_loss = float(jnp.stack(losses).mean())
    stats["val_history"].append(val_loss)

    print(f"\n=== epoch {epoch + 1} / {EPOCHS}. loss={val_loss:.4f} ===")


model = eqx.nn.inference_mode(model)
save_model(model, model_config, results_path, tag="model")

with open(f"{results_path}/stats.json", "w") as f:
    json.dump(stats, f)
