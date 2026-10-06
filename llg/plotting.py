from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SOLVER = "#eb6834"
COLORS = ("#2a78d6", "#4a3aa7", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#e34948")
MAX_PIXELS = 384  # animation frames are strided down to this many pixels a side

Curve = Float[np.ndarray, "..."]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=SECOND, labelsize=8)


def legend_below(fig: Figure, axs):
    handles = {}
    for ax in np.ravel(axs):
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))


def color(label: str, labels: list[str]) -> str:
    """The solver and the reference keep their colours; runs take the palette in order."""
    if label == "LLG solver":
        return SOLVER
    if label == "reference":
        return INK
    runs = [other for other in labels if other not in ("LLG solver", "reference")]
    return COLORS[runs.index(label) % len(COLORS)]


def panels(n_rows: int, n_cols: int, **kwargs):
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(2.9 * n_cols, 2.9 * n_rows + 0.5), squeeze=False, **kwargs
    )
    return fig, axs


def learning_curve(stats: dict) -> Figure:
    fig, axs = panels(1, 1)
    ax = axs[0, 0]
    ax.semilogy(stats["steps"], stats["train_history"], label="train")
    ax.semilogy(stats["steps"], stats["val_history"], label="validation")
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("loss", fontsize=8)
    style(ax)
    legend_below(fig, axs)
    return fig


def error_vs_step(
    curves: dict[float, dict[str, Curve]], metric: str, title: str, log: bool = True
) -> Figure:
    """curves: cell size in nm -> {label: metric per rollout step}, one panel per cell size."""
    labels = list(dict.fromkeys(label for row in curves.values() for label in row))
    fig, axs = panels(1, len(curves), sharex=True, sharey=True)
    for ax, (cell_nm, row) in zip(axs[0], curves.items()):
        for label, curve in row.items():
            steps = np.arange(1, len(curve) + 1)
            ax.plot(steps, curve, color=color(label, labels), lw=1.8, label=label)
        ax.set_title(f"{cell_nm:g} nm cells", fontsize=9)
        ax.set_xlabel("rollout step (10 ps)", fontsize=8)
        style(ax)
    axs[0, 0].set_yscale("log" if log else "linear")
    axs[0, 0].set_ylabel(metric.upper(), fontsize=9)
    fig.suptitle(title, fontsize=10)
    legend_below(fig, axs)
    return fig


def error_vs_cell(points: dict[str, dict[str, dict[float, float]]], metric: str) -> Figure:
    """points: film -> {label: {cell size in nm: metric averaged over the rollout}}."""
    labels = list(dict.fromkeys(label for film in points.values() for label in film))
    fig, axs = panels(1, len(points), sharey=True)
    for ax, (film, rows) in zip(axs[0], points.items()):
        for label, values in rows.items():
            cells = sorted(values)
            ax.plot(
                cells,
                [values[c] for c in cells],
                color=color(label, labels),
                lw=1.8,
                marker="o",
                ms=4,
                label=label,
            )
        ax.set(xscale="log", yscale="log")
        ax.set_title(film, fontsize=9)
        ax.set_xlabel("cell size (nm)", fontsize=8)
        style(ax)
    axs[0, 0].set_ylabel(f"rollout {metric.upper()}, mean over 1 ns", fontsize=8)
    legend_below(fig, axs)
    return fig


def hysteresis_loops(loops: dict[str, dict[float, dict[str, tuple]]], title: str) -> Figure:
    """loops: sweep axis -> {cell size in nm: {label: (mu0 H in mT, <m> along the axis)}}."""
    labels = list(
        dict.fromkeys(label for row in loops.values() for arms in row.values() for label in arms)
    )
    n_cols = max(len(row) for row in loops.values())
    fig, axs = panels(len(loops), n_cols, sharex=True, sharey=True)
    for r, (axis, row) in enumerate(loops.items()):
        for ax, (cell_nm, arms) in zip(axs[r], row.items()):
            for label, (h_mt, m) in arms.items():
                ax.plot(h_mt, m, color=color(label, labels), lw=1.8, label=label)
            ax.set_title(f"{cell_nm:g} nm cells", fontsize=9)
            ax.set_xlabel(f"mu0 H_{axis} (mT)", fontsize=8)
            style(ax)
        axs[r, 0].set_ylabel(f"<m_{axis}>", fontsize=9)
    fig.suptitle(title, fontsize=10)
    legend_below(fig, axs)
    return fig


def animate(
    frames: Float[np.ndarray, "..."],
    titles: list[str],
    path: Path,
    labels: tuple[str, ...] = ("$m_x$", "$m_y$", "$m_z$"),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
):
    """A GIF of frames (T, nx, ny, C), the C channels side by side, one title per frame."""
    stride = max(1, int(np.ceil(frames.shape[1] / MAX_PIXELS)))
    frames = np.asarray(frames[:, ::stride, ::stride], dtype=np.float32)
    fig, axs = plt.subplots(1, len(labels), figsize=(2.5 * len(labels), 2.9), squeeze=False)
    images = []
    for c, ax in enumerate(axs[0]):
        images.append(
            ax.imshow(frames[0, ..., c].T, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        )
        ax.set_title(labels[c], fontsize=9)
        ax.axis("off")
    text = fig.suptitle(titles[0], fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    def update(t):
        for c, image in enumerate(images):
            image.set_data(frames[t, ..., c].T)
        text.set_text(titles[t])
        return images + [text]

    FuncAnimation(fig, update, frames=len(frames), interval=100).save(path, writer="pillow")
    plt.close(fig)


def save(fig: Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(path)
