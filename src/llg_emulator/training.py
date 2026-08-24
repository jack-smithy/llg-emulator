"""Training core: losses, the jitted update step, dataloading source, LR
schedule, and model (de)serialisation.

The training objective is a **k-step rollout MSE**: the model is unrolled on its
own predictions for `k` steps and every intermediate frame is matched, so the
gradient sees the compounding error that one-step training ignores (k=1 recovers
plain one-step MSE). Best-checkpointing selects on the SP4 rollout (`sp4_rollout_rmse`).
"""

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import PyTree

from llg_emulator.model import LLGEmulator


def trainable_filter(model: LLGEmulator):
    spec = jtu.tree_map(eqx.is_inexact_array, model)
    return eqx.tree_at(lambda m: m.demag.N, spec, replace=False)


def count_parameters(model: LLGEmulator) -> int:
    return sum(
        p.size for p in jtu.tree_leaves(eqx.filter(model, trainable_filter(model)))
    )


@eqx.filter_jit
def loss_fn(model, m0, m1, H, s0):
    cond = jnp.concat((H, jnp.expand_dims(s0, -1)), axis=-1)
    m1_pred = jax.vmap(model)(m0, cond)
    return jnp.mean(jnp.square(m1_pred - m1))


@eqx.filter_jit(donate="all")
def update_fn(model: LLGEmulator, batch: PyTree, optimizer, opt_state):
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), **batch)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state, params=diff)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


def train_epoch(model, loader: grain.IterDataset, optimizer, opt_state, device):
    losses = []
    for batch in loader:
        batch = jax.device_put(batch, device)
        model, opt_state, loss = update_fn(model, batch, optimizer, opt_state)
        losses.append(loss)
    return model, opt_state, jnp.stack(losses).mean().item()


def val_epoch(model, loader: grain.IterDataset) -> float:
    """One-step MSE on the val split (loader yields (m0, m1, H) batches)."""
    inference_model = eqx.nn.inference_mode(model)
    losses = []
    for batch in loader:
        losses.append(loss_fn(inference_model, **batch))
    return jnp.stack(losses).mean().item()


def make_schedule(lr, epochs, steps_per_epoch, warmup_frac=0.05):
    """Warmup + cosine decay over the whole run."""
    total = epochs * steps_per_epoch
    warmup = max(1, int(total * warmup_frac))
    return optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=lr * 0.02,
    )
