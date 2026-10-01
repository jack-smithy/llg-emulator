import json
from itertools import islice
from argparse import ArgumentParser
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pdequinox as pdeqx
import torch
from the_well.data import WellDataset
from tqdm import tqdm

from model import ModelConfig, build_model, save_model
from utils import SCALARS, numpy_collate, prepare_batch, loss_fn, update_fn

jax.config.update("jax_compilation_cache_dir", ".jax_cache")

parser = ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--configuration", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--learning-rate", type=float, default=5e-3)
parser.add_argument("--batch-size", type=int, default=16)
# gradient steps, not epochs: the loader cycles, every batch drawing its own pool factor
parser.add_argument("--num-steps", type=int, default=10_000)
# validate (and record the mean train loss of the window since) every this many steps
parser.add_argument("--val-every", type=int, default=1_000)
parser.add_argument("--hidden-channels", type=int, default=64)
parser.add_argument("--num-blocks", type=int, default=2)
# each batch is coarse-grained by one of these integer factors (utils.downsample)
parser.add_argument("--pool-factors", type=int, nargs="+", default=[1])
args = parser.parse_args()

path = f"datasets/{args.dataset}"
results_path = (
    Path("results-v2") / args.dataset / args.configuration / f"seed_{args.seed}"
)
results_path.mkdir(exist_ok=True, parents=True)

torch.manual_seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)
key = jr.PRNGKey(args.seed)
rng = np.random.default_rng(args.seed)


IN_FRAMES = 1  # the model is a one-step map m_t -> m_{t+1}
OUT_FRAMES = 1
BATCH_SIZE = args.batch_size
NUM_WORKERS = 4
NUM_STEPS = args.num_steps
LEARNING_RATE = args.learning_rate

stats = {}
stats["config"] = {
    "batch_size": BATCH_SIZE,
    "num_steps": NUM_STEPS,
    "val_every": args.val_every,
    "learning_rate": LEARNING_RATE,
    "in_context_n": IN_FRAMES,
    "hidden_channels": args.hidden_channels,
    "num_blocks": args.num_blocks,
    "pool_factors": args.pool_factors,
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
)

key, model_key = jr.split(key)
model = build_model(model_config, model_key)
print(f"{pdeqx.count_parameters(model)} trainable parameters")

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    # cosine decay to zero over the whole run, one schedule step per batch
    optax.adamw(
        learning_rate=optax.cosine_decay_schedule(LEARNING_RATE, decay_steps=NUM_STEPS),
        weight_decay=1e-5,
    ),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# step of each entry; train_history[i] is the mean train loss over the window ending there
stats["steps"] = []
stats["train_history"] = []
stats["val_history"] = []


def cycle(loader):
    """Iterate a DataLoader forever; every pass reshuffles."""
    while True:
        yield from loader


def validate(model):
    losses = []
    # the factors in turn, so every validation scores the same batches
    for i, batch in enumerate(val_loader):
        batch = prepare_batch(batch, args.pool_factors[i % len(args.pool_factors)])
        losses.append(loss_fn(model, *batch))
    return float(jnp.stack(losses).mean())


losses = []  # device scalars, synced once per window
batches = islice(cycle(train_loader), NUM_STEPS)
for step, batch in enumerate(tqdm(batches, total=NUM_STEPS, mininterval=10), 1):
    batch = prepare_batch(batch, int(rng.choice(args.pool_factors)))
    model, opt_state, loss = update_fn(model, batch, optimizer, opt_state)
    losses.append(loss)
    if step % args.val_every == 0 or step == NUM_STEPS:
        stats["steps"].append(step)
        stats["train_history"].append(float(jnp.stack(losses).mean()))
        inference_model = eqx.nn.inference_mode(model)
        stats["val_history"].append(validate(inference_model))
        # load with load_model(results_path / "checkpoints", key, f"step_{step}")
        save_model(
            inference_model, model_config, results_path / "checkpoints", f"step_{step}"
        )
        losses = []
        print(
            f"\n=== step {step} / {NUM_STEPS}: train {stats['train_history'][-1]:.3e}, "
            f"val {stats['val_history'][-1]:.3e} ===",
            flush=True,
        )
        with open(f"{results_path}/stats.json", "w") as f:
            json.dump(stats, f)


model = eqx.nn.inference_mode(model)
save_model(model, model_config, results_path, tag="model")

with open(f"{results_path}/stats.json", "w") as f:
    json.dump(stats, f)
