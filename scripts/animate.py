"""Animate how the rolled-out average magnetization improves over training.

For each weight checkpoint we roll out a trajectory from the same initial
state and plot the spatial-average <m_x>, <m_y>, <m_z> over time, animated
across checkpoints against a static ground-truth reference.
"""

import argparse
import tempfile
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import wandb

from llg_emulator.checkpoint import load_model
from llg_emulator.config import SP4_PATH
from llg_emulator.data import load_trajectory
from llg_emulator.experiment import TrainConfig, WandbConfig
from llg_emulator.jax_setup import configure_jax
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.wandb_io import download_model_dir, find_run

configure_jax()


def main():
    parser = argparse.ArgumentParser(
        description="Animate a run's rollout across checkpoints, pulling weights from wandb."
    )
    parser.add_argument("run_name", help="wandb run display name (e.g. lively-firefly-3)")
    parser.add_argument("--project", default=WandbConfig().project)
    parser.add_argument("--entity", default=WandbConfig().entity)
    args = parser.parse_args()

    run_meta = find_run(args.run_name, args.project, args.entity)
    cfg = TrainConfig.from_dict(dict(run_meta.config))
    key = jr.PRNGKey(cfg.seed)

    wandb.init(project=args.project, entity=args.entity, id=run_meta.id, resume="must")

    m_true, H_ext = load_trajectory(path=SP4_PATH)
    n_steps = m_true.shape[0]
    ref = jnp.mean(m_true, axis=(-2, -1))  # (T, 3)

    weights_dir = Path(download_model_dir(wandb.run))
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
    out_path = Path(tempfile.mkdtemp()) / "training.gif"
    anim.save(out_path, writer="pillow")
    plt.close(fig)
    wandb.log({"training_animation": wandb.Video(str(out_path), format="gif")})
    wandb.finish()
    print(f"logged training.gif to {args.run_name} ({len(weights_paths)} frames)")


if __name__ == "__main__":
    main()
