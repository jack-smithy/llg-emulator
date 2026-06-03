import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array
from llg_emulator.model import LLGEmulator
import grain


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


@eqx.filter_jit
def loss_fn(model: LLGEmulator, m0: Array, m1: Array, H: Array) -> Array:
    """One-step MSE — the training objective, shared by train + evaluate."""
    m1_pred = jax.vmap(model)(*(m0, H))
    return jnp.mean(jnp.square(m1 - m1_pred))


@eqx.filter_jit
def update_fn(model, m0, m1, H, optimizer, state):
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), m0, m1, H)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, state = optimizer.update(grad, state)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), state, loss


def train_epoch(model, loader: grain.IterDataset, optimizer, state):
    epoch_loss = 0.0
    batch_ctr = 0
    for batch in loader:
        model, state, loss = update_fn(model, **batch, optimizer=optimizer, state=state)
        epoch_loss += loss.item()
        batch_ctr += 1  # hacky because loader doesnt have a __len__
    return model, state, epoch_loss / batch_ctr


def val_epoch(model, loader: grain.IterDataset) -> float:
    epoch_loss = 0.0
    batch_ctr = 0
    for batch in loader:
        epoch_loss += loss_fn(model, **batch).item()
        batch_ctr += 1  # hacky because loader doesnt have a __len__
    return epoch_loss / batch_ctr
