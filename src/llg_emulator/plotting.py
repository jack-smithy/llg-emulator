import matplotlib.pyplot as plt
import numpy as np


def plot_m_means(m_avg, m_avg_pred, save_path) -> None:
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))

    assert m_avg.shape == m_avg_pred.shape

    tt = range(m_avg.shape[0])

    labels = ["$<m_x>$", "$<m_y>$", "$<m_z>$"]
    for i in range(3):
        axs[i].plot(tt, m_avg[:, i], label="true")
        axs[i].plot(tt, m_avg_pred[:, i], label="pred")
        axs[i].set_ylim(-1, 1)
        axs[i].set_title(labels[i])

    plt.savefig(save_path)
    plt.close()


def plot_learning_curve(train_history, val_history, save_path) -> None:
    plt.semilogy(train_history, label="train")
    plt.semilogy(val_history, label="val")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_rollout_metric(metric, name, save_path) -> None:
    plt.figure()
    plt.plot(np.arange(metric.shape[0]), metric)
    plt.xlabel("rollout step")
    plt.ylabel(name)
    plt.title(f"SP4 {name} vs. step")
    plt.grid(True, alpha=0.3)
    plot_path = save_path / f"rollout_{name.lower()}.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"saved {plot_path}")
