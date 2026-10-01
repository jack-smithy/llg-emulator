import dataclasses
import json
import math
import random
from argparse import ArgumentParser, BooleanOptionalAction
from itertools import islice
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pdequinox as pdeqx
import torch
import yaml
from the_well.data import WellDataset
from tqdm import tqdm

from datagen.generate_varied_field import DX
from model import ModelConfig, build_model, load_model, save_model
from physics import demag_cache, llg_cache
from utils import (
    COORDS,
    SCALARS,
    prepare_batch,
    prepare_unrolled,
    resampled_mesh,
    update_fn,
    update_unrolled_fn,
)

jax.config.update("jax_compilation_cache_dir", ".jax_cache")

parser = ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--arch", type=str, required=True)
parser.add_argument("--configuration", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--learning-rate", type=float, default=5e-3)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--num-workers", type=int, default=4)
# gradient steps, not epochs: with a per-batch pool factor a "pass over the
# dataset" is no longer a meaningful unit, so the loader just cycles
parser.add_argument("--num-steps", type=int, default=10_000)
parser.add_argument("--hidden-channels", type=int, default=64)
parser.add_argument("--num-blocks", type=int, default=4)
parser.add_argument("--num-modes", type=int, default=32)
parser.add_argument("--use-demag", action=BooleanOptionalAction, default=True)
# geometry channels: absolute um coordinates, bounded edge distances, or none
parser.add_argument("--coords", choices=COORDS, default="absolute")
# condition on log(delta / l_ex): the coordinates' spacing without their offset
parser.add_argument("--cell-size-cond", action=BooleanOptionalAction, default=False)
# learn a correction to micromagnetics run on the coarse mesh (physics.CoarseLLG)
parser.add_argument("--use-solver", action=BooleanOptionalAction, default=False)
# GroupNorm in the dilated ResNet; --no-use-norm keeps the network strictly local
parser.add_argument("--use-norm", action=BooleanOptionalAction, default=True)
# train across cell sizes: every batch is coarse-grained by a factor drawn
# uniformly from this list (any real >= 1; the mesh rounds to whole cells).
# A single value trains one fixed resolution. Each distinct factor costs one
# demag-tensor build and one jit compilation, so a finite list rather than a
# continuum.
parser.add_argument("--pool-factors", type=float, nargs="+", default=[1.0])
# train on K-step autoregressive rollouts (utils.unrolled_loss_fn) instead of
# single steps; usually as a fine-tune, starting from --init-from's weights
parser.add_argument("--unroll", type=int, default=1)
parser.add_argument(
    "--init-from", type=Path, default=None, help="a run dir whose model.eqx to load"
)
# for smoke tests (test.py): write somewhere disposable
parser.add_argument("--results-root", type=Path, default=Path("results"))
# an optional directory level under results/<dataset>/ grouping related runs
parser.add_argument("--group", type=str, default="")
args = parser.parse_args()

path = f"datasets/{args.dataset}"
results_path = (
    args.results_root
    / args.dataset
    / args.group
    / args.arch
    / args.configuration
    / f"seed_{args.seed}"
)
results_path.mkdir(exist_ok=True, parents=True)

torch.manual_seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)
key = jr.PRNGKey(args.seed)


IN_FRAMES = 1  # the model is a one-step map m_t -> m_{t+1}
OUT_FRAMES = args.unroll
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
NUM_STEPS = args.num_steps
LEARNING_RATE = args.learning_rate
LOG_EVERY = 100  # gradient steps per recorded loss window
CHECKPOINT_EVERY = 5000  # a run cut short by its time limit keeps its last one

stats = {}
stats["config"] = {
    "batch_size": BATCH_SIZE,
    "num_steps": NUM_STEPS,
    "learning_rate": LEARNING_RATE,
    "in_context_n": IN_FRAMES,
    "arch": args.arch,
    "hidden_channels": args.hidden_channels,
    "num_blocks": args.num_blocks,
    "use_demag": args.use_demag,
    "coords": args.coords,
    "cell_size_cond": args.cell_size_cond,
    "use_solver": args.use_solver,
    "use_norm": args.use_norm,
    "pool_factors": args.pool_factors,
    "unroll": args.unroll,
    "init_from": str(args.init_from) if args.init_from else None,
    "seed": args.seed,
    "configuration": args.configuration,
    "group": args.group,
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

# `conditioning` indexes constant_scalars by position
assert tuple(train_dataset.metadata.constant_scalar_names) == SCALARS

# the default collate keeps batches as torch tensors, which reach the main
# process via shared memory; prepare_batch views them as numpy for free
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    generator=generator,
    drop_last=True,
    pin_memory=False,
    persistent_workers=True,
)

# the rms one-step increment of the training set (the well's `std_delta`), so
# the network's output is O(1) and a fresh model starts near persistence
with open(f"{path}/stats.yaml") as f:
    std_delta = yaml.safe_load(f)["std_delta"]
output_scale = math.sqrt(sum(v**2 for v in std_delta.values()) / len(std_delta))

model_config = ModelConfig(
    arch=args.arch,
    hidden_channels=args.hidden_channels,
    num_blocks=args.num_blocks,
    num_modes=args.num_modes,
    use_demag=args.use_demag,
    coords=args.coords,
    cell_size_cond=args.cell_size_cond,
    use_solver=args.use_solver,
    use_norm=args.use_norm,
    output_scale=output_scale,
)

# one exact demag module per pooled training mesh, nondimensional to match
# the H/Ms convention of the conditioning; None turns the input channels off
res = train_dataset.metadata.spatial_resolution
demags = {
    k: demag_cache(*resampled_mesh(res, DX, k)) if args.use_demag else None
    for k in args.pool_factors
}
# and, for a hybrid model, micromagnetics on each of those meshes
solvers = {
    k: llg_cache(*resampled_mesh(res, DX, k)) if args.use_solver else None
    for k in args.pool_factors
}

if args.init_from is not None:
    # a fine-tune keeps the output scale its checkpoint was trained with
    meta = json.loads((args.init_from / "metadata.json").read_text())
    model_config = dataclasses.replace(model_config, output_scale=meta["output_scale"])
elif args.use_solver:
    # a hybrid's network predicts the solver's residual, m1 - S(m0), which is far
    # smaller than a whole step: scale its output by that residual's rms instead,
    # measured on a few batches drawn as training draws them (first target frame)
    rng = random.Random(args.seed + 1)
    sq = []
    for batch in islice(train_loader, 16):
        k = rng.choice(args.pool_factors)
        batch = {**batch, "output_fields": batch["output_fields"][:, :1]}
        m0, m1, _ = prepare_batch(
            batch, demags[k], k, args.coords, args.cell_size_cond, solvers[k]
        )
        sq.append(float(jnp.mean(jnp.square(m1 - m0[:, :3]))))
    model_config = dataclasses.replace(
        model_config, output_scale=math.sqrt(sum(sq) / len(sq))
    )
    print(f"solver residual rms {model_config.output_scale:.3e} (output scale)")
stats["config"]["output_scale"] = model_config.output_scale

key, model_key = jr.split(key)
model = build_model(model_config, model_key)
if args.init_from is not None:
    init_model, init_config = load_model(args.init_from, model_key, tag="model")
    assert init_config == model_config, (init_config, model_config)
    model = init_model
print(f"{pdeqx.count_parameters(model)} trainable parameters")

# linear warmup, then cosine decay to 1% of the peak
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=min(1000, NUM_STEPS // 10),
    decay_steps=NUM_STEPS,
    end_value=LEARNING_RATE * 1e-2,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=1e-5),
)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

stats["train_history"] = []  # mean train loss per LOG_EVERY-step window

# python-level draws: the factor changes array shapes, so it must be decided
# outside jit (one compilation per factor)
factor_rng = random.Random(args.seed)


def cycle(loader):
    """Iterate a DataLoader forever; every pass reshuffles."""
    while True:
        yield from loader


losses = []  # device scalars; converted once per window to avoid a per-step sync
batches = islice(cycle(train_loader), NUM_STEPS)
for step, batch in enumerate(tqdm(batches, total=NUM_STEPS, mininterval=10), 1):
    k = factor_rng.choice(args.pool_factors)
    if args.unroll == 1:
        batch = prepare_batch(
            batch, demags[k], k, args.coords, args.cell_size_cond, solvers[k]
        )
        model, opt_state, loss = update_fn(model, batch, optimizer, opt_state)
    else:
        batch = prepare_unrolled(batch, k, args.coords, args.cell_size_cond)
        model, opt_state, loss = update_unrolled_fn(
            model, batch, demags[k], optimizer, opt_state, solvers[k]
        )
    losses.append(loss)
    if step % LOG_EVERY == 0 or step == NUM_STEPS:
        window = float(jnp.stack(losses).mean())
        stats["train_history"].append(window)
        losses = []
        print(f"step {step} / {NUM_STEPS}: loss={window:.3e}", flush=True)
    if step % CHECKPOINT_EVERY == 0 or step == NUM_STEPS:
        save_model(eqx.nn.inference_mode(model), model_config, results_path, "model")
        stats["steps_done"] = step
        with open(f"{results_path}/stats.json", "w") as f:
            json.dump(stats, f)
