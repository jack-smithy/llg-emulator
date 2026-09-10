from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

import wandb
from llg_emulator.config import BENCHMARK_VARIANTS, JAX_CACHE_DIR, variant_dirs
from llg_emulator.data import LLGStepperSource, dataloader_factory
from llg_emulator.io import save_model
from llg_emulator.metrics import bulk_magnetization, bulk_rmse
from llg_emulator.model import LLGEmulator, ModelConfig, with_mesh
from llg_emulator.physics import demag_for
from llg_emulator.plotting import plot_m_means
from llg_emulator.train_config import TrainConfig
from llg_emulator.training import (
    count_parameters,
    make_schedule,
    train_epoch,
    trainable_filter,
    val_epoch,
)

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)


def _to_config_dict(args, model_config):
    """Every knob that changes the run, in one flat dict.

    `vars(args)` rather than a hand-listed subset: --cosine, --weight-decay and
    --grad-clip used to be tuned but never logged, so runs could not be told
    apart in wandb. `activation` is a function object, which is not JSON-
    serialisable, so it goes in by name.
    """
    model = asdict(model_config) | {"activation": model_config.activation.__name__}
    return vars(args) | {"model_config": model}


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--cosine", action="store_true")
    # train from the model's own one-step output (detached), not the true frame
    parser.add_argument("--pushforward", action="store_true")
    # step sizes to train on, in multiples of the 10 ps base step
    parser.add_argument("--strides", type=int, nargs="+", default=[1])
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    # directory: save_model writes <out>/<tag>.eqx + <out>/metadata.json
    parser.add_argument("--out", type=str, default="results")
    # smoke-run knobs: cap how many trajectories each split maps
    parser.add_argument("--max-train-trajectories", type=int, default=None)
    parser.add_argument("--max-val-trajectories", type=int, default=None)
    parser.add_argument("--wandb-mode", type=str, choices=("online", "disabled"))
    return parser.parse_args()


def main():
    args = _parse_args()
    seed = args.seed
    save_path = Path(args.out)
    save_path.mkdir(exist_ok=True, parents=True)

    device = jax.devices()[0]

    # Every variant dir under data/train is training data. The benchmark variants
    # (sp4, large) are held out of the val loss: each sits on its own mesh, so
    # none could share a batch with the val set -- they are rollout benchmarks
    # instead (see the checkpoint block below, and evaluate.py).
    train_dataset = LLGStepperSource(
        variant_dirs("train"),
        strides=args.strides,
        max_trajectories=args.max_train_trajectories,
    )
    val_dataset = LLGStepperSource(
        variant_dirs("val", exclude=BENCHMARK_VARIANTS),
        strides=args.strides,
        max_trajectories=args.max_val_trajectories,
    )

    train_config = TrainConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_every=args.checkpoint_every,
    )

    # The mesh comes from the data, not from a hardcoded default: the demag
    # operator is the one mesh-shaped part of the model.
    model_config = ModelConfig(
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        mesh_n=train_dataset.n,
        mesh_dx=train_dataset.dx,
    )

    wandb.init(
        project="llg-emulator",
        config=_to_config_dict(args, model_config),
        mode=args.wandb_mode,
    )

    train_loader = dataloader_factory(train_dataset, batch_size=args.batch_size)
    val_loader = dataloader_factory(val_dataset, batch_size=args.batch_size)

    wandb.summary["num_train_samples"] = len(train_dataset)
    wandb.summary["num_val_samples"] = len(val_dataset)
    wandb.summary["num_train_trajectories"] = len(train_dataset.trajs)
    wandb.summary["num_val_trajectories"] = len(val_dataset.trajs)
    wandb.summary["train_mesh"] = list(train_dataset.n)

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    demag = demag_for(
        model_config.mesh_n, model_config.mesh_dx, 1.0, model_config.demag_p
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

    best_rmse = float("inf")

    with tqdm(range(1, train_config.epochs + 1)) as bar:
        bar.set_description("loss=inf")
        for i in bar:
            model, opt_state, train_loss = train_epoch(
                model=model,
                loader=train_loader(seed=i),
                optimizer=optimizer,
                opt_state=opt_state,
                device=device,
                pushforward=args.pushforward,
            )
            # no-op while train and val share a mesh; correct if they stop to
            val_loss = val_epoch(
                model=with_mesh(model, val_dataset.n, val_dataset.dx),
                loader=val_loader(seed=i),
            )

            log_dict = {
                "train/loss": train_loss,
                "val/loss": val_loss,
            }

            if i % train_config.checkpoint_every == 0:
                # SP4 rollout: a 100-step unroll on a 100x25 mesh the model has
                # never seen, under an out-of-distribution field.
                m_mean_ref, m_mean_pred = bulk_magnetization(
                    model, train_config.sp4_path
                )
                rmse = bulk_rmse(m_mean_ref, m_mean_pred)
                log_dict["sp4/bulk_rmse"] = rmse
                # log_dict["val/rollout"] = plot_m_means_plotly(m_mean_ref, m_mean_pred)
                fig, _ = plot_m_means(m_avg=m_mean_ref, m_avg_pred=m_mean_pred)
                fig.savefig(save_path / f"rollout_epoch_{i}.png")
                plt.close(fig)

                save_model(model, model_config, save_path, tag=f"epoch_{i}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    wandb.summary["best_sp4_bulk_rmse"] = rmse
                    wandb.summary["best_epoch"] = i
                    save_model(model, model_config, save_path, tag="best")
            bar.set_description(f"val={val_loss:.4e}")
            wandb.log(log_dict, step=i)
            # the SLURM log is the record when wandb is disabled
            line = f"epoch {i:3d}  train {train_loss:.4e}  val {val_loss:.4e}"
            if "sp4/bulk_rmse" in log_dict:
                line += f"  sp4_bulk_rmse {log_dict['sp4/bulk_rmse']:.4f}"
            print(line, flush=True)

    save_model(model, model_config, save_path, tag="weights")
    print(f"done: best sp4_bulk_rmse {best_rmse:.4f}", flush=True)

    # Domain-size invariance: identical physics and applied field on an 8x wider
    # mesh. The backbone is fully convolutional, so only the demag tensor has to
    # be rebuilt (bulk_magnetization does that when the meshes disagree).
    # m_mean_ref, m_mean_pred = bulk_magnetization(model, train_config.large_scale_path)
    # rmse = float(jnp.sqrt(jnp.mean((m_mean_ref - m_mean_pred) ** 2)))
    # wandb.summary["large_scale_bulk_rmse"] = rmse
    # wandb.log(
    #     {"large_scale/rollout": plot_m_means_plotly(m_mean_ref, m_mean_pred)},
    #     step=train_config.epochs,
    # )
    # fig, _ = plot_m_means(m_mean_ref, m_mean_pred)
    # fig.savefig(save_path / "large_scale_bulk.png")
    # print(f"large-scale bulk RMSE: {rmse:.4e}")

    wandb.finish()
