"""uv run train.py configs/<recipe>.yaml --seed 0  ->  RESULTS/<dataset>/<recipe>/seed_0/"""

import json
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pdequinox as pdeqx
import yaml
from jaxtyping import Array, Float
from tqdm import tqdm

from data import Batch, prepare, run_dir, to_batch, well_loader
from llg.constants import coarse_dx
from llg.physics import LLGStepper, llg_solver, rollout_batch
from model import Emulator, ModelConfig, build_model, save_model

jax.config.update("jax_compilation_cache_dir", ".jax_cache")


@dataclass
class TrainConfig:
    model: ModelConfig
    dataset: str = "llg_field_switching"
    learning_rate: float = 3e-4
    batch_size: int = 16
    num_steps: int = 10_000
    val_every: int = 1_000
    pool_factors: tuple[int, ...] = (2, 4, 8)  # one coarse-graining factor drawn per batch
    unroll: int = 8  # steps per training window, gradients through the solver
    augment: bool = True  # a random D4 symmetry of the square film per batch


def load_train_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text())
    return TrainConfig(model=ModelConfig(**raw.pop("model")), **raw)


def loss(model: Emulator, batch: Batch, solver: LLGStepper) -> Float[Array, "..."]:
    step = eqx.Partial(model, solver=solver)
    pred = rollout_batch(step, batch.m, batch.h, batch.targets.shape[1])
    return jnp.mean((pred - batch.targets) ** 2)


@eqx.filter_jit
def train_step(model, opt_state, batch: Batch, solver: LLGStepper, optimizer):
    value, grads = eqx.filter_value_and_grad(loss)(model, batch, solver)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, value


validation_loss = eqx.filter_jit(loss)


def validate(model: Emulator, loader, solvers: dict[int, LLGStepper]) -> float:
    """Every batch at the next factor in turn, so each validation scores the same batches."""
    factors = list(solvers)
    losses = [
        validation_loss(model, prepare(to_batch(sample), k), solvers[k])
        for sample, k in zip(loader, factors * len(loader))
    ]
    return float(jnp.stack(losses).mean())


def cycle(loader):
    while True:
        yield from loader


def train(config: TrainConfig, seed: int, out: Path):
    rng = np.random.default_rng(seed)
    train_loader = well_loader(
        config.dataset,
        "train",
        config.unroll,
        config.batch_size,
        seed,
        shuffle=True,
    )

    val_loader = well_loader(config.dataset, "valid", config.unroll, config.batch_size)
    nx, ny = train_loader.dataset.metadata.spatial_resolution  # type: ignore
    solvers = {k: llg_solver((nx // k, ny // k), coarse_dx(k)) for k in config.pool_factors}

    model = build_model(config.model, jr.split(jr.PRNGKey(seed))[1])
    print(f"{pdeqx.count_parameters(model)} trainable parameters")
    schedule = optax.cosine_decay_schedule(config.learning_rate, decay_steps=config.num_steps)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-5)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    stats = {"steps": [], "train_history": [], "val_history": []}
    losses = []
    batches = islice(cycle(train_loader), config.num_steps)
    for step, sample in enumerate(tqdm(batches, total=config.num_steps, mininterval=10), 1):
        k = int(rng.choice(config.pool_factors))
        g = int(rng.integers(8)) if config.augment else None
        batch = prepare(to_batch(sample), k, g)
        model, opt_state, value = train_step(model, opt_state, batch, solvers[k], optimizer)
        losses.append(value)
        if step % config.val_every and step != config.num_steps:
            continue
        inference_model = eqx.nn.inference_mode(model)
        stats["steps"].append(step)
        stats["train_history"].append(float(jnp.stack(losses).mean()))
        stats["val_history"].append(validate(inference_model, val_loader, solvers))
        losses = []
        save_model(inference_model, out / "checkpoints" / f"step_{step}.eqx")
        (out / "stats.json").write_text(json.dumps(stats))
        print(
            f"step {step}: train {stats['train_history'][-1]:.3e}, "
            f"val {stats['val_history'][-1]:.3e}",
            flush=True,
        )
    save_model(eqx.nn.inference_mode(model), out / "model.eqx")


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    config = load_train_config(args.config)
    out = run_dir(config.dataset, f"{args.config.stem}/seed_{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    train(config, args.seed, out)


if __name__ == "__main__":
    main()
