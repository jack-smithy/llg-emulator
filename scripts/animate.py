"""Animate how the rolled-out average magnetization improves over training.

For each weight checkpoint we roll out a trajectory from the same initial
state and plot the spatial-average <m_x>, <m_y>, <m_z> over time, animated
across checkpoints against a static ground-truth reference.
"""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.animation as ani
import matplotlib.pyplot as plt

from llg_emulator.checkpoint import load_model
from llg_emulator.config import RESULTS_DIR, SP4_PATH
from llg_emulator.data import load_trajectory
from llg_emulator.experiment import TrainConfig
from llg_emulator.jax_setup import configure_jax
from llg_emulator.rollout import rollout_trajectory

configure_jax()

# point this at the run dir to animate
RUN_DIR = RESULTS_DIR / "2026-05-26_12-18-10"


def main():
    cfg = TrainConfig.from_run_dir(RUN_DIR)
    key = jr.PRNGKey(cfg.seed)

    m_true, H_ext = load_trajectory(path=SP4_PATH)
    n_steps = m_true.shape[0]
    ref = jnp.mean(m_true, axis=(-2, -1))  # (T, 3)

    weights_dir = RUN_DIR / "checkpoints" / "weights"
    weights_paths = sorted(
        weights_dir.glob("weights_epoch_*.eqx"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    if not weights_paths:
        raise FileNotFoundError(f"no checkpoints found in {weights_dir}")
    epochs = [int(p.stem.split("_")[-1]) for p in weights_paths]

    m_means = []
    for weights_path in weights_paths:
        model = load_model(key=key, weights_path=weights_path, model_config=cfg.model)
        m_pred = rollout_trajectory(model, m_true, H_ext, include_init=True)
        m_means.append(jnp.mean(m_pred, axis=(-2, -1)))
    m_means = jnp.stack(m_means, axis=0)  # (num_checkpoints, T, 3)

    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
    tt = jnp.arange(n_steps)
    labels = ["$<m_x>$", "$<m_y>$", "$<m_z>$"]

    pred_lines = []
    for i in range(3):
        ax = axs[i]
        ax.plot(tt, ref[:, i], label="true", linestyle="--")
        (pred_line,) = ax.plot(tt, m_means[0, :, i], label="pred")
        pred_lines.append(pred_line)
        ax.set_ylim(-1, 1)
        ax.set_xlim(tt.min(), tt.max())
        ax.set_xlabel("rollout step")
        ax.set_title(labels[i])
    axs[2].legend()

    # epoch counter inside the middle panel so it can't be clipped
    frame_text = axs[1].text(
        0.5,
        0.92,
        "",
        transform=axs[1].transAxes,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    def update(frame):
        for i in range(3):
            pred_lines[i].set_ydata(m_means[frame, :, i])
        frame_text.set_text(f"epoch = {epochs[frame]}")
        return (*pred_lines, frame_text)

    anim = ani.FuncAnimation(
        fig, update, frames=len(weights_paths), interval=150, blit=True
    )
    out_path = RUN_DIR / "training.gif"
    anim.save(out_path, writer="pillow")
    plt.close(fig)
    print(f"saved {out_path} ({len(weights_paths)} frames)")


if __name__ == "__main__":
    main()
