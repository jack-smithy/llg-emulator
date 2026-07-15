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
from llg_emulator.io import save_model
from llg_emulator.metrics import bulk_magnetization
from llg_emulator.model import LLGEmulator, ModelConfig
from llg_emulator.plotting import plot_m_means_plotly
from llg_emulator.train_config import TrainConfig
from llg_emulator.physics import DemagField
from llg_emulator.training import (
    count_parameters,
    make_schedule,
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
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--out", type=str, default="weights/best.eqx")
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "disabled"))
    return parser.parse_args()


def main():
    args = _parse_args()
    seed = args.seed
    save_path = Path(args.out)
    save_path.mkdir(exist_ok=False, parents=True)

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
    )

    wandb.init(
        project="llg-emulator",
        config=_to_config_dict(seed, train_config, model_config),
        mode=args.wandb_mode,
    )

    device = jax.devices()[0]

    train_shards = dataset_dir("train", train_config.size)
    train_dataset = LLGStepperSource(train_shards, num_shards=None)

    val_shards = dataset_dir("val", train_config.size)
    val_dataset = LLGStepperSource(val_shards, num_shards=None)

    train_loader = dataloader_factory(train_dataset, config=train_config, device=device)
    val_loader = dataloader_factory(val_dataset, config=train_config, device=device)

    wandb.summary["num_train_samples"] = len(train_dataset)
    wandb.summary["num_val_samples"] = len(val_dataset)

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    demag = DemagField(
        model_config.mesh_n,
        model_config.mesh_dx,
        Ms=1.0,
        p=model_config.demag_p,
    )
    model = LLGEmulator(config=model_config, demag=demag, key=subkey)
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

    with tqdm(range(1, train_config.epochs + 1)) as bar:
        bar.set_description("loss=inf")
        for i in bar:
            model, opt_state, train_loss = train_epoch(
                model=model,
                loader=train_loader(seed=i),
                optimizer=optimizer,
                opt_state=opt_state,
                device=device,
            )
            val_loss = val_epoch(model=model, loader=val_loader(seed=i))

            log_dict = {"train/loss": train_loss, "val/loss": val_loss}

            if i % train_config.checkpoint_every == 0:
                m_mean_ref, m_mean_pred = bulk_magnetization(
                    model, train_config.sp4_path
                )
                log_dict["val/rollout"] = plot_m_means_plotly(m_mean_ref, m_mean_pred)

                save_model(model, model_config, save_path, tag=f"epoch_{i}")
            bar.set_description(f"val={val_loss:.4e}")
            wandb.log(log_dict, step=i)

    save_model(model, model_config, save_path, tag="weights")

    wandb.finish()
