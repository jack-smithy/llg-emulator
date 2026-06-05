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


@eqx.filter_jit(donate="all-except-first")
def update_fn(batch, model, optimizer, opt_state):
    m0, m1, H = batch["m0"], batch["m1"], batch["H"]
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), m0, m1, H)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


def train_epoch(model, loader: grain.IterDataset, optimizer, opt_state):
    epoch_loss = jnp.array(0.0)  # accumulate on-device; one sync at the end
    n_batches = 0
    for batch in loader:
        model, opt_state, loss = update_fn(batch, model, optimizer, opt_state)
        epoch_loss += loss
        n_batches += 1
    return model, opt_state, (epoch_loss / n_batches).item()


def val_epoch(model, loader: grain.IterDataset) -> float:
    inference_model = eqx.nn.inference_mode(model)  # True is the default
    epoch_loss = jnp.array(0.0)
    n_batches = 0
    for batch in loader:
        epoch_loss += loss_fn(inference_model, **batch)
        n_batches += 1
    return (epoch_loss / n_batches).item()
