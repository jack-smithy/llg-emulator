import argparse
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import wandb

from llg_emulator.checkpoint import load_model
from llg_emulator.config import SP4_PATH, dataset_dir
from llg_emulator.data import load_trajectory, LLGStepperSource, dataloader_factory
from llg_emulator.experiment import TrainConfig, WandbConfig
from llg_emulator.jax_setup import configure_jax
from llg_emulator.metrics import correlation, nRMSE
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import loss_fn
from llg_emulator.plotting import plot_m_means, plot_rollout_metric
from llg_emulator.wandb_io import download_model_dir, find_run
from tqdm import tqdm

configure_jax()


def rollout_stats(model, sample_path: Path):
    """Roll out a trajectory; return per-step nRMSE/corr curves, overall
    correlation, and the trajectory's normalized dt (for labelling)."""
    m_true, H_ext = load_trajectory(sample_path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    nrmse_curve = jax.vmap(nRMSE)(m_pred, m_true)
    corr_curve = jax.vmap(correlation)(m_pred, m_true)
    return nrmse_curve, corr_curve, correlation(m_pred, m_true)


def plot_trajectory_means(model, sample_path: Path):
    """Roll out a trajectory and plot true vs. predicted <m_x,y,z> spatial
    means over time (the same view produced during training)."""
    m_true, H_ext = load_trajectory(sample_path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    return plot_m_means(
        m_avg=jnp.mean(m_true, axis=(2, 3)),
        m_avg_pred=jnp.mean(m_pred, axis=(2, 3)),
    )


def correlations_trajectory(model, path: Path):
    paths_list = list(path.iterdir())

    traj_corrs = []
    traj_rmses = []
    for p in tqdm(paths_list):
        m_true, H_ext = load_trajectory(p)
        m_pred = rollout_trajectory(model, m_true, H_ext)
        traj_corrs.append(jax.vmap(correlation)(m_pred, m_true))
        traj_rmses.append(jax.vmap(nRMSE)(m_pred, m_true))

    traj_corrs = jnp.stack(traj_corrs, axis=0)
    traj_rmses = jnp.stack(traj_rmses, axis=0)

    corrs_mean = jnp.mean(traj_corrs, axis=0)
    rmses_mean = jnp.mean(traj_rmses, axis=0)

    corrs_std = jnp.std(traj_corrs, axis=0)
    rmses_std = jnp.std(traj_rmses, axis=0)

    steps = corrs_mean.shape[0]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axs[0].plot(range(steps), corrs_mean)
    axs[0].fill_between(
        range(steps),
        corrs_mean - corrs_std,
        corrs_mean + corrs_std,
        alpha=0.3,
    )
    axs[0].set_ylim((0, 1))
    axs[0].set(title="Correlation", xlabel="step", ylabel="correlation")

    axs[1].plot(range(steps), rmses_mean)
    axs[1].fill_between(
        range(steps),
        rmses_mean - rmses_std,
        rmses_mean + rmses_std,
        alpha=0.3,
    )
    axs[1].set_ylim((0, 1))
    axs[1].set(title="nRMSE", xlabel="step", ylabel="nRMSE")

    return fig, axs


def dataset_loss(model, batch_size: int, split: str, size: str, seed=0) -> float:
    """Mean one-step MSE over a data split."""
    dataset = LLGStepperSource(dataset_dir(split, size))
    loader = dataloader_factory(dataset, batch_size=batch_size)
    losses = []
    for batch in loader(seed=seed):
        losses.append(loss_fn(model, **batch))
    return jnp.stack(losses).mean().item()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained run, pulling weights from wandb."
    )
    parser.add_argument(
        "run_name", help="wandb run display name (e.g. lively-firefly-3)"
    )
    parser.add_argument("--project", default=WandbConfig().project)
    parser.add_argument("--entity", default=WandbConfig().entity)
    args = parser.parse_args()

    print("starting evaluation")
    run_meta = find_run(args.run_name, args.project, args.entity)
    cfg = TrainConfig.from_dict(dict(run_meta.config))
    key = jr.PRNGKey(cfg.seed)

    # resume the run so eval metrics/plots land back on it
    wandb.init(project=args.project, entity=args.entity, id=run_meta.id, resume="must")
    weights_dir = Path(download_model_dir(wandb.run))
    model = load_model(
        key=key,
        weights_path=weights_dir / "model.eqx",
        model_config=cfg.model,
    )
    print("model loaded")

    fig_rollout, axs = correlations_trajectory(model, dataset_dir("val", "med"))

    train_loss = dataset_loss(
        model,
        split="train",
        batch_size=cfg.data.batch_size,
        size=cfg.data.size,
    )
    val_loss = dataset_loss(
        model,
        split="val",
        batch_size=cfg.data.batch_size,
        size=cfg.data.size,
    )
    print("calculated dataset loss")

    nrmse_curve, corr_curve, corr = rollout_stats(model, SP4_PATH)
    print("rollout trajectory done\n\n")

    stats = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "rollout_nrmse_mean": float(jnp.mean(nrmse_curve)),
        "rollout_nrmse_final": float(nrmse_curve[-1]),
        "rollout_correlation": float(corr),
    }
    for k, v in stats.items():
        print(f"{k} = {v:.6e}")
    wandb.summary.update(stats)

    fig_rmse, axs = plot_rollout_metric(nrmse_curve, "nRMSE")
    fig_corr, axs = plot_rollout_metric(corr_curve, "Correlation")
    fig_traj, axs = plot_trajectory_means(model, SP4_PATH)
    wandb.log(
        {
            "plots/eval/rollouts": wandb.Image(fig_rollout),
            "plots/sp4/rollouts": wandb.Image(fig_traj),
            "plots/sp4/rollout_nrmse": wandb.Image(fig_rmse),
            "plots/sp4/rollout_correlation": wandb.Image(fig_corr),
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
