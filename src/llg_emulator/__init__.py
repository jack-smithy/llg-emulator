"""llg-emulator: learn LLG magnetization dynamics for a thin film."""

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import optax
from grain.sources import ArrayRecordDataSource
from tqdm import tqdm

import wandb
from llg_emulator.config import JAX_CACHE_DIR, dataset_dir
from llg_emulator.data import LLGStepperSource, TrajectoryDataSource, dataloader_factory
from llg_emulator.metrics import bulk_magnetization
from llg_emulator.model import LLGEmulator, ModelConfig
from llg_emulator.plotting import plot_m_means_plotly
from llg_emulator.train_config import TrainConfig
from llg_emulator.training import (
    count_parameters,
    train_epoch,
    trainable_filter,
    val_epoch,
)

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)


def _to_config_dict(seed, train_config, model_config):
    return {
        "seed": seed,
        "train_config": asdict(train_config),
        "model_config": asdict(model_config),
    }


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument(
        "--wandb-mode", type=str, choices=("online", "disabled"), default="online"
    )
    parser.add_argument("--cpu-buffer-size", type=int, default=8)
    parser.add_argument("--device-buffer-size", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    return parser.parse_args()


def get_shards(path: Path):
    return [str(p) for p in path.glob("*.arrayrecord")]


def main():
    args = _parse_args()

    seed = args.seed

    model_config = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
    )

    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        size=args.size,
        checkpoint_every=args.checkpoint_every,
        cpu_buffer_size=args.cpu_buffer_size,
        device_buffer_size=args.device_buffer_size,
    )

    config = _to_config_dict(
        seed=seed,
        train_config=train_config,
        model_config=model_config,
    )

    wandb.init(
        project="llg-emulator",
        config=config,
        mode=args.wandb_mode,
    )

    device = jax.devices()[0]

    train_source = ArrayRecordDataSource(
        get_shards(dataset_dir("train", train_config.size))
    )

    val_source = ArrayRecordDataSource(
        get_shards(dataset_dir("val", train_config.size))
    )

    sp4_source = ArrayRecordDataSource(
        get_shards(dataset_dir("sp4", train_config.size))
    )

    train_dataset = LLGStepperSource(train_source)
    val_dataset = LLGStepperSource(val_source)

    sp4_trajectory = TrajectoryDataSource(sp4_source)

    train_loader = dataloader_factory(train_dataset, config=train_config, device=device)
    val_loader = dataloader_factory(val_dataset, config=train_config, device=device)

    train_ratio = len(train_dataset) / (len(val_dataset) + len(train_dataset))
    wandb.summary["num_train_samples"] = len(train_dataset)
    wandb.summary["num_val_samples"] = len(val_dataset)
    wandb.summary["train_ratio"] = train_ratio

    key = jr.PRNGKey(seed)

    key, subkey = jr.split(key)
    model = LLGEmulator(config=model_config, key=subkey)
    wandb.summary["num_parameters"] = count_parameters(model)

    optimizer = optax.adam(learning_rate=train_config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, trainable_filter(model)))

    with tqdm(range(train_config.epochs)) as bar:
        bar.set_description("loss=inf")
        for i in bar:
            model, opt_state, train_loss = train_epoch(
                model=model,
                loader=train_loader(seed=i),
                optimizer=optimizer,
                opt_state=opt_state,
            )

            val_loss = val_epoch(model=model, loader=val_loader(seed=i))

            log_dict = {
                "train/loss": train_loss,
                "val/loss": val_loss,
            }

            if i % train_config.checkpoint_every == 0:
                m_mean_ref, m_mean_pred = bulk_magnetization(model, sp4_trajectory)
                log_dict["val/rollout"] = plot_m_means_plotly(m_mean_ref, m_mean_pred)

                # corr_mean, corr_std = correlation_epoch(model, val_dataset)
                # log_dict["val/corr_rollout"] = plot_corr_plotly(
                #     corr_mean=corr_mean, corr_std=corr_std
                # )
                # log_dict["val/corr_mean"] = corr_mean.mean().item()
                # log_dict["val/corr_std"] = corr_std.mean().item()
                # log_dict["val/corr_final"] = corr_mean[-1].item()

            bar.set_description(f"loss={val_loss:.4e}")
            wandb.log(
                log_dict,
                step=i,
            )

    wandb.log(
        {
            "val/rollout": plot_m_means_plotly(
                *bulk_magnetization(model, sp4_trajectory)
            )
        },
        step=train_config.epochs,
    )

    wandb.finish()
