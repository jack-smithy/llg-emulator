from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from tqdm import tqdm

from llg_emulator.config import JAX_CACHE_DIR, dataset_dir
from llg_emulator.data import LLGStepperSource
from llg_emulator.io import load_model
from llg_emulator.metrics import correlation
from llg_emulator.model import ModelConfig
from llg_emulator.physics import DemagField
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import count_parameters

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--model-path", type=str, required=True)
    return parser.parse_args()


def rollout_correlation(model, dataset: LLGStepperSource):
    total = len(dataset.trajs)
    corrs = []
    for m_true, H_ext in tqdm(zip(dataset.trajs, dataset.fields), total=total):
        m_pred = rollout_trajectory(model, m_true=m_true, H_ext=H_ext)
        corrs.append(correlation(pred=m_pred, ref=m_true)[0])
    corrs = jnp.stack(corrs, axis=0)
    return corrs.mean(axis=0), corrs.std(axis=0)


def plot_rollout_correlation(cm, cs):
    xx = range(cm.shape[0])
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    axs.plot(xx, cm)
    axs.fill_between(xx, cm + cs, cm - cs, alpha=0.4)
    axs.set_xlabel("step")
    axs.set_ylabel("correlation")
    return fig, axs


def main():
    args = _parse_args()
    seed = args.seed

    model_path = Path(args.model_path)

    device = jax.devices()[0]

    val_shards = dataset_dir("val", args.size)

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    mesh = ModelConfig()
    demag = DemagField(mesh.mesh_n, mesh.mesh_dx, Ms=1.0, p=mesh.demag_p)
    model = load_model(path=model_path, demag=demag, tag="weights", key=subkey)

    val_dataset = LLGStepperSource(val_shards, num_shards=None, strides=[1])
    print(count_parameters(model), len(val_dataset), device)
    cm, cs = rollout_correlation(model, val_dataset)
    fig, axs = plot_rollout_correlation(cm=cm, cs=cs)
    fig.savefig(model_path / "correlation.png")


if __name__ == "__main__":
    main()
