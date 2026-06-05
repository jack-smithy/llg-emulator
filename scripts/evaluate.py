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

configure_jax()
# point this at the run dir to evaluate
RUN_DIR = RESULTS_DIR / "baseline"


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


def rollout_stats(model, sample_path: Path):
    """Roll out the sp4 trajectory; return per-step nRMSE curve + correlation."""
    m_true, H_ext = load_trajectory(sample_path)
    m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
    nrmse_curve = jax.vmap(nRMSE)(m_pred, m_true)
    return nrmse_curve, correlation(m_pred, m_true)


def main():
    print("starting evaluation")
    cfg = TrainConfig.from_run_dir(RUN_DIR)
    key = jr.PRNGKey(cfg.seed)
    model = load_model(
        key=key, weights_path=RUN_DIR / "weights.eqx", model_config=cfg.model
    )
    print("model loaded")

    train_loss = dataset_loss(model, "train", cfg.data.size)
    val_loss = dataset_loss(model, "val", cfg.data.size)
    print("calculated dataset loss")

    nrmse_curve, corr = rollout_stats(model, SP4_PATH)
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

    plt.figure()
    plt.plot(jnp.arange(nrmse_curve.shape[0]), nrmse_curve)
    plt.xlabel("rollout step")
    plt.ylabel("nRMSE")
    plt.title("sp4 rollout error vs. step")
    plt.grid(True, alpha=0.3)
    plot_path = RUN_DIR / "rollout_nrmse.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
