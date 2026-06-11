import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree

from llg_emulator.model import LLGEmulator
from llg_emulator.metrics import MSE, correlation
from llg_emulator.rollout import rollout_trajectories, rollout_trajectory
from llg_emulator.data import load_trajectory


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
    return MSE(m1_pred, m1)


@eqx.filter_jit(donate="all-except-first")
def evaluate_fn(model: LLGEmulator, batch: PyTree, model_sharding, data_sharding):
    model = eqx.filter_shard(model, model_sharding)
    batch = eqx.filter_shard(batch, data_sharding)
    return loss_fn(model, **batch)


@eqx.filter_jit(donate="all")
def update_fn(
    model: LLGEmulator,
    batch: PyTree,
    optimizer,
    opt_state,
    model_sharding,
    data_sharding,
):
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
    batch = eqx.filter_shard(batch, data_sharding)

    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), **batch)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state)
    diff = eqx.apply_updates(diff, updates)
    model = eqx.combine(diff, static)
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)
    return model, opt_state, loss


def train_epoch(
    model,
    loader: grain.IterDataset,
    optimizer,
    opt_state,
    model_sharding,
    data_sharding,
):
    model, opt_state = eqx.filter_shard((model, opt_state), model_sharding)

    losses = []
    for batch in loader:
        batch = eqx.filter_shard(batch, data_sharding)
        model, opt_state, loss = update_fn(
            batch=batch,
            data_sharding=data_sharding,
            model=model,
            model_sharding=model_sharding,
            optimizer=optimizer,
            opt_state=opt_state,
        )
        losses.append(loss)
    return model, opt_state, jnp.stack(losses).mean().item()


def val_epoch(model, loader: grain.IterDataset, model_sharding, data_sharding) -> float:
    inference_model = eqx.nn.inference_mode(model)

    losses = []
    for batch in loader:
        loss = evaluate_fn(
            batch=batch,
            model=inference_model,
            model_sharding=model_sharding,
            data_sharding=data_sharding,
        )
        losses.append(loss)
    return jnp.stack(losses).mean().item()


@eqx.filter_jit
def rollout_sharded(model, m_ref, h_ext, model_sharding, data_sharding):
    model = eqx.filter_shard(model, model_sharding)
    m_ref = eqx.filter_shard(m_ref, data_sharding)
    h_ext = eqx.filter_shard(h_ext, data_sharding)
    m_pred = rollout_trajectories(m_ref, h_ext, model)
    return eqx.filter_shard(m_pred, data_sharding)


def correlation_epoch(model, source, model_sharding, data_sharding):
    m_ref = jnp.stack(source.trajs, axis=0)
    h_ext = jnp.stack(source.fields, axis=0)

    # Shard the trajectory rollout across devices on the leading (batch) axis;
    # that axis must divide the device count, so pad up and slice back before
    # the (batch-reducing) correlation so padded trajectories don't bias it.
    n = m_ref.shape[0]
    num_devices = data_sharding.mesh.size
    pad = (-n) % num_devices
    if pad:
        m_ref = jnp.concatenate([m_ref, m_ref[:pad]], axis=0)
        h_ext = jnp.concatenate([h_ext, h_ext[:pad]], axis=0)

    m_pred = rollout_sharded(model, m_ref, h_ext, model_sharding, data_sharding)
    mean, std = correlation(m_pred[:n], m_ref[:n])
    return mean, std


def bulk_magnetization(model, path):
    m_true, H_ext = load_trajectory(path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    return jnp.mean(m_true, axis=(2, 3)), jnp.mean(m_pred, axis=(2, 3))
