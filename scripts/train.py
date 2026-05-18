import os

from llg_emulator.config import DEVICE

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

from llg_emulator.jax_setup import configure_jax

configure_jax()

from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import optax
from tqdm import tqdm

from llg_emulator.checkpoint import make_run_dir
from llg_emulator.config import SIZE, dataset_dir
from llg_emulator.data import JaxLoader, LLGDataset, load_trajectory
from llg_emulator.model import LLGEmulator
from llg_emulator.plotting import plot_learning_curve, plot_m_means
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import (
    count_parameters,
    train_epoch,
    val_epoch,
)


def test_rollout(model, m_true, H_ext, save_path: Path) -> None:
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    plot_m_means(
        m_avg=jnp.mean(m_true, axis=(2, 3)),
        m_avg_pred=jnp.mean(m_pred, axis=(2, 3)),
        save_path=save_path,
    )


def main():
    save_path = make_run_dir()
    print(f"run dir = {save_path}")

    train_dataset = LLGDataset(dataset_dir("train"), warmup_steps=1)
    val_dataset = LLGDataset(dataset_dir("val"), warmup_steps=1)

    train_loader = JaxLoader(
        dataset=train_dataset, batch_size=128, shuffle=True, pin_memory=True
    )
    val_loader = JaxLoader(
        dataset=val_dataset, batch_size=128, shuffle=True, pin_memory=True
    )

    train_ratio = len(train_dataset) / (len(val_dataset) + len(train_dataset))
    print(f"num train samples = {len(train_dataset)}")
    print(f"num val samples = {len(val_dataset)}")
    print(f"train ratio = {train_ratio * 100:.2f}%")

    key = jr.PRNGKey(0)
    key, subkey = jr.split(key)
    model = LLGEmulator(key=subkey)
    print(f"num parameters = {count_parameters(model)}")

    # rollout viz trajectory + initial checkpoint
    m_true, H_ext = load_trajectory(dataset_dir("val") / "sample_292")
    test_rollout(
        model,
        m_true=m_true,
        H_ext=H_ext,
        save_path=save_path / "checkpoints/trjs/trj_epoch_0.png",
    )
    eqx.tree_serialise_leaves(
        save_path / "checkpoints/weights/weights_epoch_0.eqx", model
    )

    optimizer = optax.adam(1e-3)
    state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_history = []
    val_history = []
    with tqdm(range(10)) as bar:
        for i in bar:
            model, state, train_loss = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                state=state,
            )
            train_history.append(train_loss)

            val_loss = val_epoch(model=model, loader=val_loader)
            val_history.append(val_loss)
            bar.set_description(f"loss={val_loss:.4e}")

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
