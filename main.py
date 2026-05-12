import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
from tqdm import tqdm

from loader import JaxLoader, LLGDataset
from model import LLGEmulator
from plot import plot_learning_curve, plot_m_means
from utils import load_trajectory, rollout, stepper_fn

jax.config.update("jax_compilation_cache_dir", ".jax_cache")


def count_parameters(model: eqx.Module):
    return sum(p.size for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


@eqx.filter_jit
def loss_fn(model, m0, m1):
    m1_pred = jax.vmap(model)(m0)
    return jnp.mean(jnp.square(m1 - m1_pred))


@eqx.filter_jit
def update_fn(model, m0, m1, optimizer, state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, m0, m1)
    updates, state = optimizer.update(grad, state)
    model = eqx.apply_updates(model, updates)
    return model, state, loss


def train_epoch(model, loader, optimizer, state):
    epoch_train_loss = 0
    for batch in loader:
        model, state, loss = update_fn(
            model,
            *batch,
            optimizer=optimizer,
            state=state,
        )
        epoch_train_loss += loss.item()
    return model, state, epoch_train_loss / len(loader)


def val_epoch(model, loader):
    epoch_val_loss = 0
    for batch in loader:
        loss = loss_fn(model, *batch)
        epoch_val_loss += loss.item()
    return epoch_val_loss / len(loader)


def test_rollout(model, m_true, H_ext, save_path):
    m_pred = rollout(
        lambda x: stepper_fn(x, model, H_ext),
        n=99,
        include_init=True,
    )(m_true[0])

    m_avg = jnp.mean(m_true, axis=(2, 3))
    m_avg_pred = jnp.mean(m_pred, axis=(2, 3))

    plot_m_means(
        m_avg=m_avg,
        m_avg_pred=m_avg_pred,
        save_path=save_path,
    )


def main():
    save_path = Path("results")
    train_dataset = LLGDataset(
        "../micromagnetic-data/data/v2/small/train",
        warmup_steps=1,
    )
    val_dataset = LLGDataset(
        "../micromagnetic-data/data/v2/small/val",
        warmup_steps=1,
    )

    train_loader = JaxLoader(
        dataset=train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = JaxLoader(
        dataset=val_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
    )

    train_ratio = len(train_dataset) / (len(val_dataset) + len(train_dataset))
    print(f"num train samples = {len(train_dataset)}")
    print(f"num val samples = {len(val_dataset)}")
    print(f"train ratio = {train_ratio * 100:.2f}%")

    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)

    model = LLGEmulator(key=subkey)

    print(f"num parameters = {count_parameters(model)}")

    # load rollout data and do init trajectory
    m_true, H_ext = load_trajectory(
        Path("../micromagnetic-data/data/v2/small/train/sample_1")
    )
    test_rollout(
        model,
        m_true=m_true,
        H_ext=H_ext,
        save_path=save_path / "checkpoints/trjs/trj_init.png",
    )

    optimizer = optax.adam(1e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_history = []
    val_history = []
    with tqdm(range(256)) as bar:
        for i in bar:
            # train pass
            model, state, train_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                state=state,
            )
            train_history.append(train_loss)

            # val pass
            val_loss = val_epoch(
                model=model,
                loader=val_loader,
            )
            val_history.append(val_loss)
            bar.set_description(f"loss={val_loss:.4e}")

            # test rollout and save weights
            if (i + 1) % 8 == 0:
                test_rollout(
                    model,
                    m_true=m_true,
                    H_ext=H_ext,
                    save_path=save_path / f"checkpoints/trjs/trj_epoch_{i + 1}.png",
                )
                eqx.tree_serialise_leaves(
                    save_path / f"checkpoints/weights/weights_epoch_{i + 1}.eqx",
                    model,
                )

    eqx.tree_serialise_leaves(save_path / "weights.eqx", model)

    plot_learning_curve(
        train_history=train_history,
        val_history=val_history,
        save_path=save_path / "learning_curve.png",
    )


if __name__ == "__main__":
    main()
