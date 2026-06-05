import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from llg_emulator.checkpoint import load_model
from llg_emulator.config import RESULTS_DIR, SP4_PATH, dataset_dir
from llg_emulator.data import load_trajectory, LLGStepperSource, dataloader_factory
from llg_emulator.experiment import TrainConfig
from llg_emulator.jax_setup import configure_jax
from llg_emulator.metrics import correlation, nRMSE
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import loss_fn
from llg_emulator.plotting import plot_m_means, plot_rollout_metric
from tqdm import tqdm

configure_jax()
# point this at the run dir to evaluate
RUN_DIR = RESULTS_DIR / "2026-06-05_13-33-41"


def rollout_stats(model, sample_path: Path):
    """Roll out a trajectory; return per-step nRMSE/corr curves, overall
    correlation, and the trajectory's normalized dt (for labelling)."""
    m_true, H_ext = load_trajectory(sample_path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    nrmse_curve = jax.vmap(nRMSE)(m_pred, m_true)
    corr_curve = jax.vmap(correlation)(m_pred, m_true)
    return nrmse_curve, corr_curve, correlation(m_pred, m_true)


def plot_trajectory_means(model, sample_path: Path, save_path: Path) -> None:
    """Roll out a trajectory and plot true vs. predicted <m_x,y,z> spatial
    means over time (the same view produced during training)."""
    m_true, H_ext = load_trajectory(sample_path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    plot_m_means(
        m_avg=jnp.mean(m_true, axis=(2, 3)),
        m_avg_pred=jnp.mean(m_pred, axis=(2, 3)),
        save_path=save_path / "traj.png",
    )


def correlations_trajectory(model, path: Path, save_path):
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

    fig.savefig(save_path / "rollouts.png")
    plt.close(fig)


def dataset_loss(model, split: str, size: str, seed=0) -> float:
    """Mean one-step MSE over a data split."""
    dataset = LLGStepperSource(dataset_dir(split, size))
    loader = dataloader_factory(dataset, batch_size=128)
    total = 0.0
    batch_ctr = 0
    for batch in loader(seed=seed):
        total += loss_fn(model, **batch).item()
        batch_ctr += 1
    return total / batch_ctr


def main():
    print("starting evaluation")
    cfg = TrainConfig.from_run_dir(RUN_DIR)
    key = jr.PRNGKey(cfg.seed)
    model = load_model(
        key=key,
        weights_path=RUN_DIR / "weights.eqx",
        model_config=cfg.model,
    )
    print("model loaded")

    correlations_trajectory(model, dataset_dir("val", "med"), RUN_DIR)

    train_loss = dataset_loss(model, "train", cfg.data.size)
    val_loss = dataset_loss(model, "val", cfg.data.size)
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

    metrics_path = RUN_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"saved {metrics_path}")

    plot_rollout_metric(nrmse_curve, "nRMSE", RUN_DIR)
    plot_rollout_metric(corr_curve, "Correlation", RUN_DIR)
    plot_trajectory_means(model, SP4_PATH, RUN_DIR)


if __name__ == "__main__":
    main()
