import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from pathlib import Path
from test import load_trajectory
from einops import repeat

path = Path("../micromagnetic-data/data/v2/low_res/train/sample_1")
m, _ = load_trajectory(path=path)
ref = np.mean(m, axis=(-2, -1))
print(ref.shape)  # (100, 3)

data = np.load("training_evolution.npy")
print(data.shape)  # (64, 100, 3)


def loss_fn(ref, data):
    ref = repeat(ref, "n c -> t n c", t=data.shape[0])
    return np.mean(np.square(data - ref), axis=(1, 2))


plt.plot(loss_fn(ref, data))
plt.savefig("sp4_loss_curve.png")
plt.close()

fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
tt = np.arange(1e-11, 1e-9, 1e-11)

labels = ["$<m_x>$", "$<m_y>$", "$<m_z>$"]

pred_lines = []
ref_lines = []


for i in range(3):
    ax = axs[i]

    # static reference (draw once)
    (ref_line,) = ax.plot(tt, ref[:, i], label="true", linestyle="--")
    ref_lines.append(ref_line)

    # animated prediction
    (pred_line,) = ax.plot(tt, data[0, :, i], label="pred")
    pred_lines.append(pred_line)

    ax.set_ylim(-1, 1)
    ax.set_xlim(tt.min(), tt.max())
    ax.set_title(labels[i])

axs[2].legend()

frame_text = fig.text(0.5, 1, "", ha="center", va="top")

epochs = range(64)


def update(frame):
    for i in range(3):
        pred_lines[i].set_ydata(data[frame, :, i])
    frame_text.set_text(f"epoch={epochs[frame]}")
    return pred_lines  # only animated artists


anim = ani.FuncAnimation(
    fig,
    update,
    frames=64,
    interval=75,  # ms between frames
    blit=True,
)

anim.save("trainig.gif", writer="pillow")
