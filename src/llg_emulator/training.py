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
def pushforward_loss_fn(model: LLGEmulator, m0: Array, m2: Array, H: Array) -> Array:
    """Two-step pushforward MSE — the *training* objective.

    Step 1 is rolled out under stop_gradient so the network sees its own
    prediction as input; only step 2 carries gradients.
    """
    m1_pred = jax.lax.stop_gradient(jax.vmap(model)(m0, H))  # pushforward, no grad
    m2_pred = jax.vmap(model)(m1_pred, H)  # differentiated step
    return jnp.mean(jnp.square(m2 - m2_pred))


@eqx.filter_jit(donate="all-except-first")
def update_fn(batch, model, optimizer, opt_state):
    m0, m2, H = batch["m0"], batch["m2"], batch["H"]
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return pushforward_loss_fn(eqx.combine(diff, static), m0, m2, H)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


def train_epoch(model, loader: grain.IterDataset, optimizer, opt_state):
    losses = []
    for batch in loader:
        model, opt_state, loss = update_fn(batch, model, optimizer, opt_state)
        losses.append(loss)  # device scalars, no sync
    return model, opt_state, jnp.stack(losses).mean().item()


def val_epoch(model, loader: grain.IterDataset) -> float:
    inference_model = eqx.nn.inference_mode(model)
    losses = []
    for batch in loader:
        loss = loss_fn(inference_model, batch["m0"], batch["m1"], batch["H"])
        losses.append(loss)
    return jnp.stack(losses).mean().item()
