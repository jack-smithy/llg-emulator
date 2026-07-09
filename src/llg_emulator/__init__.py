"""llg-emulator: learn LLG magnetization dynamics for a thin film."""

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import optax
from tqdm import tqdm

import wandb
from llg_emulator.config import JAX_CACHE_DIR, dataset_dir
from llg_emulator.data import LLGStepperSource, dataloader_factory
from llg_emulator.metrics import bulk_magnetization, correlation_epoch
from llg_emulator.model import LLGEmulator, ModelConfig
from llg_emulator.plotting import plot_corr_plotly, plot_m_means_plotly
from llg_emulator.train_config import TrainConfig
from llg_emulator.training import (
    RolloutSource,
    count_parameters,
    make_schedule,
    save_model,
    sp4_rollout_rmse,
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    # winning recipe knobs (see BENCHMARKS.md / CLAUDE.md)
    parser.add_argument(
        "--rollout-k", type=int, default=4, help="unroll length in loss"
    )
    parser.add_argument("--cosine", action="store_true", help="warmup+cosine schedule")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--augment", action="store_true", help="D4xZ2 symmetry aug (16x)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="weights/best_dt.eqx",
        help="where to save the best-SP4-rollout checkpoint",
    )
    parser.add_argument(
        "--wandb-mode", type=str, choices=("online", "disabled"), default="online"
    )
    parser.add_argument("--cpu-buffer-size", type=int, default=8)
    parser.add_argument("--device-buffer-size", type=int, default=4)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="epochs between SP4 rollout eval + best-checkpoint save",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    seed = args.seed

    model_config = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        cond_dim=3,
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

    wandb.init(
        project="llg-emulator",
        config=_to_config_dict(seed, train_config, model_config),
        mode=args.wandb_mode,
    )

    device = jax.devices()[0]

    train_dataset = LLGStepperSource(dataset_dir("train", train_config.size))
    val_dataset = LLGStepperSource(dataset_dir("val", train_config.size))

    train_loader = dataloader_factory(train_dataset, config=train_config, device=device)
    val_loader = dataloader_factory(val_dataset, config=train_config, device=device)

    wandb.summary["num_train_samples"] = len(train_dataset)
    wandb.summary["num_val_samples"] = len(val_dataset)

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    model = LLGEmulator(config=model_config, key=subkey)
    wandb.summary["num_parameters"] = count_parameters(model)

    steps_per_epoch = max(1, len(train_dataset) // train_config.batch_size)
    lr = (
        make_schedule(args.learning_rate, args.epochs, steps_per_epoch)
        if args.cosine
        else args.learning_rate
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(learning_rate=lr, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, trainable_filter(model)))

    best_rmse = float("inf")
    with tqdm(range(train_config.epochs)) as bar:
        bar.set_description("loss=inf")
        for i in bar:
            model, opt_state, train_loss = train_epoch(
                model=model,
                loader=train_loader(seed=i),
                optimizer=optimizer,
                opt_state=opt_state,
                k=args.rollout_k,
            )
            val_loss = val_epoch(model=model, loader=val_loader(seed=i))

            log_dict = {"train/loss": train_loss, "val/loss": val_loss}

            if (i + 1) % train_config.checkpoint_every == 0:
                # SP4 rollout: log the plot + best-checkpoint on the bulk RMSE
                m_mean_ref, m_mean_pred = bulk_magnetization(
                    model, train_config.sp4_path
                )
                log_dict["val/rollout"] = plot_m_means_plotly(m_mean_ref, m_mean_pred)

                corr_mean, corr_std = correlation_epoch(model, val_dataset)
                log_dict["val/corr_rollout"] = plot_corr_plotly(
                    corr_mean=corr_mean, corr_std=corr_std
                )
                log_dict["val/corr_mean"] = corr_mean.mean().item()
                log_dict["val/corr_final"] = corr_mean[-1].item()

                # stride-1 is the checkpoint metric (comparable to the fixed-dt
                # baseline); also log big-step rollouts — the point of the feature.
                rmse = sp4_rollout_rmse(model, train_config.sp4_path, stride=1)
                log_dict["val/sp4_bulk_rmse"] = rmse
                for s in (5, 10):
                    log_dict[f"val/sp4_bulk_rmse@{s}"] = sp4_rollout_rmse(
                        model, train_config.sp4_path, stride=s
                    )
                if rmse < best_rmse:
                    best_rmse = rmse
                    save_model(model, model_config, Path(args.out))
                    wandb.summary["best_sp4_bulk_rmse"] = best_rmse
                    wandb.summary["best_epoch"] = i

            bar.set_description(f"val={val_loss:.4e} sp4_best={best_rmse:.4f}")
            wandb.log(log_dict, step=i)

    # if checkpoint-every never fired, still leave a reloadable model behind
    if best_rmse == float("inf"):
        save_model(model, model_config, Path(args.out))
    print(f"best SP4 bulk RMSE {best_rmse:.4f} -> {args.out}")
    wandb.finish()
