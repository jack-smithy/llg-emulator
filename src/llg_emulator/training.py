import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array

from llg_emulator.model import LLGEmulator
from llg_emulator.metrics import mse


def trainable_filter(model: LLGEmulator):
    """Bool pytree: True for trainable leaves, False for the fixed demag tensor.

    The demag submodule is a physical operator with no learnable parameters,
    but its precomputed tensor `demag.N` is a large array leaf that the default
    is_array filter would otherwise feed to the optimiser. Freeze it.
    """
    spec = jtu.tree_map(eqx.is_inexact_array, model)
    return eqx.tree_at(lambda m: m.demag.N, spec, replace=False)


def count_parameters(model: LLGEmulator) -> int:
    return sum(
        p.size for p in jtu.tree_leaves(eqx.filter(model, trainable_filter(model)))
    )


def loss_fn(model: LLGEmulator, m0: Array, m1: Array, H: Array) -> Array:
    """One-step MSE — the training objective, shared by train + evaluate."""
    m1_pred = jax.vmap(model)(*(m0, H))
    return mse(m1_pred, m1)


@eqx.filter_jit(donate="all-except-first")
def evaluate_fn(batch, model):
    return loss_fn(model, **batch)


@eqx.filter_jit(donate="all-except-first")
def update_fn(batch, model, optimizer, opt_state):
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), **batch)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


def train_epoch(model, loader: grain.IterDataset, optimizer, opt_state):
    losses = []
    for batch in loader:
        model, opt_state, loss = update_fn(batch, model, optimizer, opt_state)
        losses.append(loss)
    return model, opt_state, jnp.stack(losses).mean().item()


def val_epoch(model, loader: grain.IterDataset) -> float:
    inference_model = eqx.nn.inference_mode(model)
    losses = []
    for batch in loader:
        loss = evaluate_fn(batch, inference_model)
        losses.append(loss)
    return jnp.stack(losses).mean().item()
