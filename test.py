import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from tqdm import tqdm

from loader import JaxLoader, LLGDataset
from model import LLGEmulator
from utils import load_trajectory, rollout, stepper_fn


def load_model(key, weights_path):
    model = LLGEmulator(key=key)
    model = eqx.tree_deserialise_leaves(weights_path, model)
    return model


def nRMSE(pred, ref):
    diff_norm = jnp.linalg.norm(pred - ref)
    ref_norm = jnp.linalg.norm(ref)

    return diff_norm / ref_norm


def correlation(pred, ref):
    pred_normalized = pred / jnp.linalg.norm(pred)
    ref_normalized = ref / jnp.linalg.norm(ref)

    return jnp.dot(pred_normalized.flatten(), ref_normalized.flatten())


def main():

    key = jr.PRNGKey(0)
    path = Path("../micromagnetic-data/data/v2/small/val")

    refs = []
    for sample_path in tqdm(path.iterdir()):
        m_ref, H_ext = load_trajectory(sample_path)
        refs.append((m_ref, H_ext))

    test_trjs = jnp.stack([r[0] for r in refs], axis=0)
    test_fields = jnp.stack([r[1] for r in refs], axis=0)

    print(test_trjs.shape, test_fields.shape)

    model = load_model(key=key, weights_path="results/weights.eqx")

    def single_rollout(init_state, field):
        rollout_fn = rollout(
            lambda x: stepper_fn(x, model, field),
            n=99,
            include_init=True,
        )
        return rollout_fn(init_state)

    init_states = test_trjs[:, 0]  # shape: (batch, ...) — grab first step for all trjs

    pred_trjs = jax.vmap(single_rollout)(init_states, test_fields)

    jnp.save("results/pred.npy", pred_trjs)
    jnp.save("results/ref.npy", test_trjs)


if __name__ == "__main__":
    main()
