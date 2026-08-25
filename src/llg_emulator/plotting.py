import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_m_means_plotly(m_avg, m_avg_pred):
    assert m_avg.shape == m_avg_pred.shape

    m_avg = np.asarray(m_avg)
    m_avg_pred = np.asarray(m_avg_pred)

    tt = np.arange(m_avg.shape[0])
    labels = ["⟨m_x⟩", "⟨m_y⟩", "⟨m_z⟩"]

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=labels)

    for i in range(3):
        col = i + 1
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=m_avg[:, i],
                name="true",
                line={"color": "#636efa"},
                legendgroup="true",
                showlegend=(i == 0),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=tt,
                y=m_avg_pred[:, i],
                name="pred",
                line={"color": "#ef553b"},
                legendgroup="pred",
                showlegend=(i == 0),
            ),
            row=1,
            col=col,
        )
        fig.update_yaxes(range=[-1, 1], row=1, col=col)

    return fig


def plot_corr_plotly(corr_mean, corr_std):
    x = np.arange(corr_mean.shape[0])

    fig = go.Figure()

    # Mean line
    fig.add_trace(
        go.Scatter(
            x=x,
            y=corr_mean,
            mode="lines",
            name="mean",
        )
    )

    # Shaded ± std region
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([corr_mean + corr_std, (corr_mean - corr_std)[::-1]]),
            fill="toself",
            fillcolor="rgba(0,100,80,0.3)",
            line={"color": "rgba(255,255,255,0)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_yaxes(range=[0, 1.1])

    return fig


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
