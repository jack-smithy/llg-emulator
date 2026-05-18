import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array


def count_parameters(model: eqx.Module) -> int:
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


@eqx.filter_jit
def loss_fn(model: eqx.Module, m0: Array, m1: Array) -> Array:
    """One-step MSE — the training objective, shared by train + evaluate."""
    m1_pred = jax.vmap(model)(m0)
    return jnp.mean(jnp.square(m1 - m1_pred))


@eqx.filter_jit
def update_fn(model, m0, m1, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, m0, m1)
    updates, state = optimizer.update(grad, state)
    model = eqx.apply_updates(model, updates)
    return model, state, loss


def train_epoch(model, loader, optimizer, state):
    epoch_loss = 0.0
    for batch in loader:
        model, state, loss = update_fn(
            model, *batch, optimizer=optimizer, state=state
        )
        epoch_loss += loss.item()
    return model, state, epoch_loss / len(loader)


def val_epoch(model, loader) -> float:
    epoch_loss = 0.0
    for batch in loader:
        epoch_loss += loss_fn(model, *batch).item()
    return epoch_loss / len(loader)
