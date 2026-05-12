import matplotlib.pyplot as plt
import jax.numpy as jnp


def plot_m_means(m_avg, m_avg_pred, save_path):
    fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
    tt = jnp.arange(1e-11, 1e-9, 1e-11)

    labels = ["$<m_x>$", "$<m_y>$", "$<m_z>$"]
    for i in range(3):
        axs[i].plot(tt, m_avg[:, i], label="true")
        axs[i].plot(tt, m_avg_pred[:, i], label="pred")
        axs[i].set_ylim(-1, 1)
        axs[i].set_xlim(0, 1e-9)
        axs[i].set_title(labels[i])

    plt.savefig(save_path)
    plt.close()


def plot_learning_curve(train_history, val_history, save_path):
    plt.semilogy(train_history, label="train")
    plt.semilogy(val_history, label="val")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
