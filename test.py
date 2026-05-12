import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


import json
import glob
from pathlib import Path

import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from einops import rearrange, repeat

from model import LLGEmulator


def rollout(
    stepper_fn,
    n,
    *,
    include_init=False,
):
    """
    Transform an autonomous timestepper into a function that efficiently unrolls
    a trajectory.
    """

    def scan_fn(u, _):
        u_next = stepper_fn(u)
        return u_next, u_next

    def rollout_fn(init):
        _, history = jax.lax.scan(
            scan_fn,
            init,
            None,
            length=n,
        )

        if include_init:
            return jnp.concatenate(
                [
                    jnp.expand_dims(init, axis=0),
                    history,
                ],
                axis=0,
            )
        else:
            return history

    return rollout_fn


def load_trajectory(path: Path):
    m = np.load(path / "m.npy")
    m = m[1:, :, :, 0, :]
    m = rearrange(m, "t w h c -> t c w h")
    with open(path / "params.json", "r") as f:
        p = json.load(f)
    H_ext = np.asarray(p["H_ext"])
    Ms = p["material"]["Ms"]
    H_ext_nondimensionalized = H_ext / Ms

    return m, H_ext_nondimensionalized


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


def prepare_inputs(m, H_ext):
    H_ext = repeat(H_ext, "c -> c h w", h=m.shape[1], w=m.shape[2])
    feature = jnp.concatenate((m, H_ext), axis=0)
    return feature


def load_model(key, weights_path):
    model = LLGEmulator(
        num_modes=32,
        num_blocks=4,
        hidden_channels=64,
        key=key,
    )
    model = eqx.tree_deserialise_leaves(weights_path, model)
    return model


def stepper_fn(m, model, H_ext):
    H_ext = repeat(H_ext, "c -> c h w", h=m.shape[1], w=m.shape[2])
    feature = jnp.concatenate((m, H_ext), axis=0)
    return model(feature)


def main():

    key = jr.PRNGKey(0)
    path = Path("../micromagnetic-data/data/v2/sp4/sample_0")
    m, H_ext = load_trajectory(path=path)

    weights_paths = glob.glob("weights_epoch*.eqx")
    sorted_weights_paths = sorted(
        weights_paths, key=lambda p: int(p.split("_")[-1].replace(".eqx", ""))
    )

    m_means = []
    for weights_path in sorted_weights_paths:
        model = load_model(key, weights_path)

        m_pred = rollout(
            lambda x: stepper_fn(x, model, H_ext),
            n=99,
            include_init=True,
        )(m[0])
        m_means.append(jnp.mean(m_pred, axis=(-2, -1)))

    m_means = jnp.stack(m_means, axis=0)
    print(m_means.shape)
    jnp.save("training_evolution.npy", m_means)

    # plot_m_means(m, m_pred)


if __name__ == "__main__":
    main()
