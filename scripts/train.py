import argparse
from dataclasses import asdict

import equinox as eqx
import jax
import jax.random as jr
import jax.sharding as jshard
from tqdm import tqdm

import wandb
from llg_emulator.config import SP4_PATH, dataset_dir
from llg_emulator.data import LLGStepperSource, dataloader_factory
from llg_emulator.experiment import TrainConfig, build_activation, build_optimizer
from llg_emulator.jax_setup import configure_jax
from llg_emulator.model import LLGEmulator
from llg_emulator.plotting import plot_m_means_plotly, plot_corr_plotly
from llg_emulator.training import (
    count_parameters,
    train_epoch,
    trainable_filter,
    val_epoch,
    correlation_epoch,
    bulk_magnetization,
)
from llg_emulator.wandb_io import save_weights

configure_jax()


def main(cfg):

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

            log_dict = {
                "train/loss": train_loss,
                "val/loss": val_loss,
            }

            if i % cfg.checkpoint_every == 0:
                m_mean_ref, m_mean_pred = bulk_magnetization(model, SP4_PATH)
                log_dict["val/rollout"] = plot_m_means_plotly(m_mean_ref, m_mean_pred)

                corr_mean, corr_std = correlation_epoch(
                    model,
                    val_dataset,
                    model_sharding=model_sharding,
                    data_sharding=data_sharding,
                )
                log_dict["val/corr_rollout"] = plot_corr_plotly(
                    corr_mean=corr_mean, corr_std=corr_std
                )
                log_dict["val/corr_mean"] = corr_mean.mean().item()
                log_dict["val/corr_std"] = corr_std.mean().item()
                log_dict["val/corr_final"] = corr_mean[-1].item()

                save_weights(model, step=i)

            bar.set_description(f"loss={val_loss:.4e}")
            wandb.log(
                log_dict,
                step=i,
            )

    wandb.log(
        {"val/rollout": plot_m_means_plotly(*bulk_magnetization(model, SP4_PATH))},
        step=cfg.epochs,
    )

    save_weights(model, step=cfg.epochs, aliases="final")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LLG emulator.")
    parser.add_argument(
        "--config",
        default="configs/default.toml",
        help="path to a TOML training config (fully specifies the run)",
    )
    args = parser.parse_args()

    cfg = TrainConfig.from_toml(args.config)

    main(cfg=cfg)
