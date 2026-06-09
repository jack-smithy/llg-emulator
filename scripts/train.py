import argparse
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import wandb
from tqdm import tqdm

from llg_emulator.checkpoint import make_run_dir
from llg_emulator.config import dataset_dir
from llg_emulator.data import load_trajectory, dataloader_factory, LLGStepperSource
from llg_emulator.experiment import TrainConfig, build_activation, build_optimizer
from llg_emulator.jax_setup import configure_jax
from llg_emulator.model import LLGEmulator
from llg_emulator.plotting import plot_learning_curve, plot_m_means
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import (
    count_parameters,
    train_epoch,
    trainable_filter,
    val_epoch,
)

configure_jax()


def test_rollout(model, m_true, H_ext, save_path: Path) -> None:
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    plot_m_means(
        m_avg=jnp.mean(m_true, axis=(2, 3)),
        m_avg_pred=jnp.mean(m_pred, axis=(2, 3)),
        save_path=save_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train the LLG emulator.")
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="path to a TOML training config (fully specifies the run)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_toml(args.config)
    save_path = make_run_dir()
    cfg.save(save_path, args.config)

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name or save_path.name,
        dir=str(save_path),
        config=asdict(cfg),
    )
    wandb.summary["run_dir"] = str(save_path)
    wandb.summary["config_path"] = args.config

    train_dataset = LLGStepperSource(dataset_dir("train", cfg.data.size))
    val_dataset = LLGStepperSource(dataset_dir("val", cfg.data.size))

    train_loader = dataloader_factory(
        train_dataset,
        batch_size=cfg.data.batch_size,
        prefetch=2 * cfg.data.batch_size,
    )
    val_loader = dataloader_factory(
        val_dataset,
        batch_size=cfg.data.batch_size,
        prefetch=2 * cfg.data.batch_size,
    )

    train_ratio = len(train_dataset) / (len(val_dataset) + len(train_dataset))
    wandb.summary["num_train_samples"] = len(train_dataset)
    wandb.summary["num_val_samples"] = len(val_dataset)
    wandb.summary["train_ratio"] = train_ratio

    key = jr.PRNGKey(cfg.seed)
    key, subkey = jr.split(key)
    model = LLGEmulator(
        hidden_channels=cfg.model.hidden_channels,
        num_blocks=cfg.model.num_blocks,
        activation=build_activation(cfg.model.activation),
        mesh_n=cfg.model.mesh_n,
        mesh_dx=cfg.model.mesh_dx,
        demag_p=cfg.model.demag_p,
        key=subkey,
    )
    wandb.summary["num_parameters"] = count_parameters(model)

    # rollout viz trajectory + initial checkpoint
    viz_path = dataset_dir("train", cfg.data.size).parent / cfg.data.viz_sample
    m_true, H_ext = load_trajectory(viz_path)
    trj0_path = save_path / "checkpoints/trjs/trj_epoch_0.png"
    test_rollout(model, m_true=m_true, H_ext=H_ext, save_path=trj0_path)
    wandb.log({"rollout/trj": wandb.Image(str(trj0_path))}, step=0)
    eqx.tree_serialise_leaves(
        save_path / "checkpoints/weights/weights_epoch_0.eqx", model
    )

    optimizer = build_optimizer(cfg.optim)
    opt_state = optimizer.init(eqx.filter(model, trainable_filter(model)))

    train_history = []
    val_history = []
    with tqdm(range(cfg.epochs)) as bar:
        for i in bar:
            model, opt_state, train_loss = train_epoch(
                model=model,
                loader=train_loader(seed=i),
                optimizer=optimizer,
                opt_state=opt_state,
            )
            train_history.append(train_loss)

            val_loss = val_epoch(model=model, loader=val_loader(i))
            val_history.append(val_loss)
            bar.set_description(f"loss={val_loss:.4e}")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss}, step=i)

            if (i + 1) % cfg.checkpoint_every == 0:
                trj_path = save_path / f"checkpoints/trjs/trj_epoch_{i + 1}.png"
                test_rollout(model, m_true=m_true, H_ext=H_ext, save_path=trj_path)
                wandb.log({"rollout/trj": wandb.Image(str(trj_path))}, step=i)
                eqx.tree_serialise_leaves(
                    save_path / f"checkpoints/weights/weights_epoch_{i + 1}.eqx",
                    model,
                )

    eqx.tree_serialise_leaves(save_path / "weights.eqx", model)
    curve_path = save_path / "learning_curve.png"
    plot_learning_curve(
        train_history=train_history,
        val_history=val_history,
        save_path=curve_path,
    )
    wandb.log({"learning_curve": wandb.Image(str(curve_path))})
    wandb.finish()


if __name__ == "__main__":
    main()
