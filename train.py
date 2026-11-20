"""uv run train.py configs/<recipe>.yaml --seed 0  ->  RESULTS/<dataset>/<recipe>/seed_0/

Two objectives: `rollout` (the default) fits `unroll`-step rollouts of the coarse model to
the coarse-grained truth, gradients through the solver; `tendency` fits the closure field
alone, so that the coarse dm/dt matches the block mean of the 5 nm one on single snapshots
(force matching, no ODE solves). Closure formulations only, see `tendency_loss`.

A run resumes from its last checkpoint, so a long recipe can be a chain of short jobs:
`--max-minutes` stops a link at a checkpoint, and the next link picks it up.
"""

import json
import pickle
import shutil
import time
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

from data import Batch, block_mean, d4, prepare, run_dir, to_batch, well_loader
from llg.constants import DT, DX, coarse_dx
from llg.physics import LLGStepper, llg_solver, rollout_batch
from model import Emulator, ModelConfig, build_model, load_model, save_model

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
    objective: str = "rollout"  # or "tendency"
    init_from: str | None = None  # start from this run's model.eqx (<recipe>/seed_<s>)
    warmup_steps: int = 0  # linear from zero; for init_from, whose optimiser starts cold


def load_train_config(path: Path) -> TrainConfig:
    raw = yaml.safe_load(path.read_text())
    return TrainConfig(model=ModelConfig(**raw.pop("model")), **raw)


def rollout_loss(model: Emulator, batch: Batch, solver: LLGStepper, fine: LLGStepper, k: int):
    step = eqx.Partial(model, solver=solver)
    pred = rollout_batch(step, batch.m, batch.h, batch.targets.shape[1])
    return jnp.mean((pred - batch.targets) ** 2)


def tendency_loss(model: Emulator, batch: Batch, solver: LLGStepper, fine: LLGStepper, k: int):
    """batch.m on the 5 nm mesh. The coarse state is the block mean m_bar / |m_bar|; its
    true rate of change is the block mean of the fine dm/dt, projected onto its tangent
    plane and divided by |m_bar|. Error in units of the change over one frame (DT)."""

    def one(m_fine, h):
        h_fine = jnp.broadcast_to(h, m_fine.shape)
        m_bar, dm_bar = block_mean(m_fine, k), block_mean(fine.rhs(m_fine, h_fine), k)
        norm = jnp.linalg.norm(m_bar, axis=-1, keepdims=True)
        m = m_bar / norm
        target = (dm_bar - (m * dm_bar).sum(-1, keepdims=True) * m) / norm
        h_coarse = jnp.broadcast_to(h, m.shape) + model.closure_field(m, h, solver)  # type: ignore
        return jnp.mean((DT * (solver.rhs(m, h_coarse) - target)) ** 2)

    return jnp.mean(jax.vmap(one)(batch.m, batch.h))


LOSSES = {"rollout": rollout_loss, "tendency": tendency_loss}


@eqx.filter_jit
def train_step(model, opt_state, batch, solver, fine, k, optimizer, objective):
    value, grads = eqx.filter_value_and_grad(LOSSES[objective])(model, batch, solver, fine, k)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state, value


@eqx.filter_jit
def validation_loss(model, batch, solver, fine, k, objective):
    return LOSSES[objective](model, batch, solver, fine, k)


def prepare_for(objective: str, sample: dict, k: int, g: int | None) -> Batch:
    if objective == "rollout":
        return prepare(to_batch(sample), k, g)
    batch = to_batch(sample)  # tendency: augmented, left on the fine mesh
    m, h = jnp.asarray(batch.m), jnp.asarray(batch.h)
    if g is not None:
        m, h = d4(m, h, g)
    return Batch(m, jnp.zeros(0), h)


def validate(model, loader, solvers: dict, fine: LLGStepper, objective: str) -> float:
    """Every batch at the next factor in turn, so each validation scores the same batches."""
    factors = list(solvers)
    losses = [
        validation_loss(model, prepare_for(objective, sample, k, None), solvers[k], fine, k, objective)
        for sample, k in zip(loader, factors * len(loader))
    ]
    return float(jnp.stack(losses).mean())


def cycle(loader):
    while True:
        yield from loader


def save_state(out: Path, model, opt_state, step: int, stats: dict, rng):
    eqx.tree_serialise_leaves(out / "state.eqx", (model, opt_state))
    (out / "state.pkl").write_bytes(pickle.dumps({"step": step, "stats": stats, "rng": rng}))


def train(config: TrainConfig, seed: int, out: Path, max_minutes: float | None):
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    n_frames = config.unroll if config.objective == "rollout" else 1
    train_loader = well_loader(config.dataset, "train", n_frames, config.batch_size, seed, shuffle=True)
    val_loader = well_loader(config.dataset, "valid", n_frames, config.batch_size)
    nx, ny = train_loader.dataset.metadata.spatial_resolution  # type: ignore
    solvers = {k: llg_solver((nx // k, ny // k), coarse_dx(k)) for k in config.pool_factors}
    fine = llg_solver((nx, ny), DX)

    if config.init_from:
        model = eqx.nn.inference_mode(load_model(run_dir(config.dataset, config.init_from)), False)
    else:
        model = build_model(config.model, jr.split(jr.PRNGKey(seed))[1])
    print(f"{pdeqx.count_parameters(model)} trainable parameters")
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, config.learning_rate, config.warmup_steps, config.num_steps
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-5)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    first, stats = 1, {"steps": [], "train_history": [], "val_history": [], "seconds": []}
    if (out / "state.pkl").exists():
        saved = pickle.loads((out / "state.pkl").read_bytes())
        model, opt_state = eqx.tree_deserialise_leaves(out / "state.eqx", (model, opt_state))
        first, stats, rng = saved["step"] + 1, saved["stats"], saved["rng"]
        train_loader = well_loader(  # a fresh shuffle per link
            config.dataset, "train", n_frames, config.batch_size, seed + first, shuffle=True
        )
        print(f"resumed at step {first}")
    if first > config.num_steps:
        return
    if first == 1:  # the starting point: the coarse solver itself, for zero-init models
        stats["val_at_start"] = validate(
            eqx.nn.inference_mode(model), val_loader, solvers, fine, config.objective
        )
        print(f"step 0: val {stats['val_at_start']:.3e}", flush=True)

    losses = []
    batches = islice(cycle(train_loader), config.num_steps - first + 1)
    for step, sample in enumerate(tqdm(batches, total=config.num_steps - first + 1, mininterval=10), first):
        k = int(rng.choice(config.pool_factors))
        g = int(rng.integers(8)) if config.augment else None
        batch = prepare_for(config.objective, sample, k, g)
        model, opt_state, value = train_step(
            model, opt_state, batch, solvers[k], fine, k, optimizer, config.objective
        )
        losses.append(value)
        if step % config.val_every and step != config.num_steps:
            continue
        inference_model = eqx.nn.inference_mode(model)
        stats["steps"].append(step)
        stats["train_history"].append(float(jnp.stack(losses).mean()))
        stats["val_history"].append(validate(inference_model, val_loader, solvers, fine, config.objective))
        stats["seconds"].append(time.monotonic() - start)
        losses = []
        save_model(inference_model, out / "checkpoints" / f"step_{step}.eqx")
        save_state(out, model, opt_state, step, stats, rng)
        (out / "stats.json").write_text(json.dumps(stats))
        print(
            f"step {step}: train {stats['train_history'][-1]:.3e}, "
            f"val {stats['val_history'][-1]:.3e}",
            flush=True,
        )
        if max_minutes and step != config.num_steps:
            per_block = (time.monotonic() - start) / (step - first + 1) * config.val_every
            if time.monotonic() - start + 1.2 * per_block > 60 * max_minutes:
                print(f"stopping at step {step} for the time limit; resume to continue")
                return
    save_model(eqx.nn.inference_mode(model), out / "model.eqx")


def main():
    parser = ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-minutes", type=float, default=None, help="stop at a checkpoint")
    args = parser.parse_args()

    config = load_train_config(args.config)
    out = run_dir(config.dataset, f"{args.config.stem}/seed_{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")
    train(config, args.seed, out, args.max_minutes)


if __name__ == "__main__":
    main()
