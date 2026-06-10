import argparse
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as jshard
from tqdm import tqdm

import wandb
from llg_emulator.config import SP4_PATH, dataset_dir
from llg_emulator.data import LLGStepperSource, dataloader_factory, load_trajectory
from llg_emulator.experiment import TrainConfig, build_activation, build_optimizer
from llg_emulator.jax_setup import configure_jax
from llg_emulator.metrics import correlation
from llg_emulator.model import LLGEmulator
from llg_emulator.plotting import plot_m_means_plotly
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import (
    count_parameters,
    train_epoch,
    trainable_filter,
    val_epoch,
)
from llg_emulator.wandb_io import save_weights

configure_jax()


def plot_trajectory_means(model, sample_path: Path):
    """Roll out a trajectory and plot true vs. predicted <m_x,y,z> spatial
    means over time (the same view produced during training)."""
    m_true, H_ext = load_trajectory(sample_path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    return plot_m_means_plotly(
        m_avg=jnp.mean(m_true, axis=(2, 3)),
        m_avg_pred=jnp.mean(m_pred, axis=(2, 3)),
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

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=cfg.wandb.name,
        config=asdict(cfg),
    )
    assert wandb.run is not None

    num_devices = len(jax.devices())
    mesh = jax.make_mesh(
        (num_devices,), ("batch",), axis_types=(jax.sharding.AxisType.Auto,)
    )
    data_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec("batch"))
    model_sharding = jshard.NamedSharding(mesh, jshard.PartitionSpec())

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

    optimizer = build_optimizer(cfg.optim)
    opt_state = optimizer.init(eqx.filter(model, trainable_filter(model)))

    with tqdm(range(cfg.epochs)) as bar:
        bar.set_description("loss=inf")
        for i in bar:
            model, opt_state, train_loss = train_epoch(
                model=model,
                loader=train_loader(seed=i),
                optimizer=optimizer,
                opt_state=opt_state,
                model_sharding=model_sharding,
                data_sharding=data_sharding,
            )

            val_loss = val_epoch(
                model=model,
                loader=val_loader(seed=i),
                model_sharding=model_sharding,
                data_sharding=data_sharding,
            )

            corr = correlation(model, val_dataset)

            # if i % cfg.checkpoint_every == 0:
            #     log_dict["val/rollout"] = plot_trajectory_means(
            #         model=model, sample_path=SP4_PATH
            #     )
            #     save_weights(model, step=i)
            bar.set_description(f"loss={val_loss:.4e}")
            wandb.log(
                {"train/loss": train_loss, "val/loss": val_loss, "val/corr": corr},
                step=i,
            )

    wandb.log(
        {"val/rollout": plot_trajectory_means(model=model, sample_path=SP4_PATH)},
        step=cfg.epochs,
    )

    save_weights(model, step=cfg.epochs, aliases="final")
    wandb.finish()


if __name__ == "__main__":
    main()
