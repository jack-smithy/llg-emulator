import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_learning_curves(stats):
    train_history = stats["train_history"]
    val_history = stats["val_history"]
    steps = np.arange(1, len(train_history) + 1)

    fig, ax = plt.subplots(figsize=(5, 3), sharey=True)
    ax.semilogy(steps, train_history, label="train")
    ax.semilogy(steps, val_history, label="val")
    ax.grid(True, alpha=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig, ax


def plot_rollout(ref_bulk, pred_bulk):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3), sharey=True)
    labels = ["$<m_x>$", "$<m_y>$", "$<m_z>$"]

    tt = np.arange(0, ref_bulk.shape[0])

    for i, ax in enumerate(axs):
        ax.plot(tt, ref_bulk[..., i], label="ref")
        ax.plot(tt, pred_bulk[..., i], "--", label="pred")
        ax.set_title(labels[i])
        ax.set_ylim((-1, 1))
        ax.set_ylabel("m")
        ax.set_xlabel("t")
        ax.grid(alpha=0.5)
        ax.set_xlim(left=0, right=tt[-1])
    axs[2].legend()
    fig.tight_layout()
    return fig, axs


def plot_norms(ref_norm, pred_norm):

    tt = np.arange(0, ref_norm.shape[0])
    fig, ax = plt.subplots(figsize=(5, 3), sharey=True)
    ax.plot(tt, ref_norm)
    ax.plot(tt, pred_norm)
    ax.set_ylabel("$|m_{pred}|$")
    ax.set_xlabel("t")
    ax.grid(alpha=0.5)
    ax.set_xlim(left=0, right=tt[-1])
    fig.tight_layout()
    return fig, ax


def animate_channels(arr, interval=100, cmap="RdBu_r"):
    """arr: (T, C, H, W). Returns a FuncAnimation."""
    arr = np.moveaxis(arr, -1, 1)
    T, C = arr.shape[:2]
    fig, axes = plt.subplots(1, C, figsize=(2 * C, 3.2), squeeze=False)
    axes = axes.ravel()

    labels = ["$m_x$", "$m_y$", "$m_z$"]

    ims = []
    for c, ax in enumerate(axes):
        im = ax.imshow(arr[0, c], cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(labels[c])
        ax.axis("off")
        ims.append(im)

    def update(t):
        for c, im in enumerate(ims):
            im.set_data(arr[t, c])
        # fig.suptitle(f"t = {t}")
        return ims

    return FuncAnimation(fig, update, frames=T, interval=interval, blit=False)
