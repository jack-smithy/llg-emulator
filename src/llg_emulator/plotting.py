import matplotlib.pyplot as plt
import numpy as np


def plot_m_means(m_avg, m_avg_pred):
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))

    assert m_avg.shape == m_avg_pred.shape

    tt = range(m_avg.shape[0])

    labels = ["$<m_x>$", "$<m_y>$", "$<m_z>$"]
    for i in range(3):
        axs[i].plot(tt, m_avg[:, i], label="true")
        axs[i].plot(tt, m_avg_pred[:, i], label="pred")
        axs[i].set_ylim(-1, 1)
        axs[i].set_title(labels[i])

    return fig, axs


def plot_learning_curve(train_history, val_history):
    fig, axs = plt.subplots(1, 1, figsize=(5, 4))
    axs.semilogy(train_history, label="train")
    axs.semilogy(val_history, label="val")
    axs.legend()
    return fig, axs


def plot_rollout_metric(metric, name):
    fig, axs = plt.subplots()
    axs.plot(np.arange(metric.shape[0]), metric)
    axs.set_xlabel("rollout step")
    axs.set_ylabel(name)
    axs.set_title(f"SP4 {name} vs. step")
    axs.grid(True, alpha=0.3)
    return fig, axs
