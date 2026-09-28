"""Glue between `the_well`'s loaders and the JAX model.

`WellDataset` serves channel-last frames, `(B, T, Lx, Ly, F)`, and the model
takes one channel-first frame per sample, `(3, Lx, Ly)`, plus a conditioning
vector `[Hx, Hy] / Ms` (see `training.build_model`). Everything here is the
layout change and the batching around it.
"""

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
from einops import rearrange
from torch.utils.data import default_collate
from tqdm import tqdm

# `batch["constant_scalars"]` in the order the generator writes them
# (generate_varied_field.create_split); train.py checks the dataset agrees.
SCALARS = ("Ms", "A", "alpha", "Hx", "Hy", "Hz")


def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data samples into a batch.
    default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
    """
    return jtu.tree_map(np.asarray, default_collate(batch))


def conditioning(scalars):
    """`constant_scalars` (B, 6) -> `cond` (B, 2): `[Hx, Hy] / Ms`.

    Hz is always zero in the data and Ms is constant, so this is the whole
    per-sample variation, nondimensionalised.
    """
    return scalars[:, 3:5] / scalars[:, :1]


def prepare_batch(batch):
    """One-step batch -> `(m0, m1, cond)` in the model's layout, all numpy.

    m0, m1: (B, 3, Lx, Ly); cond: (B, 2).
    """
    m0 = rearrange(batch["input_fields"], "B 1 Lx Ly F -> B F Lx Ly")
    m1 = rearrange(batch["output_fields"], "B 1 Lx Ly F -> B F Lx Ly")
    return m0, m1, conditioning(batch["constant_scalars"])


@eqx.filter_jit
def predict(model, m0, cond):
    """One step for a batch: (B, 3, Lx, Ly), (B, 2) -> (B, 3, Lx, Ly)."""
    return jax.vmap(model)(m0, cond)


@eqx.filter_jit
def rollout(model, x, cond, n_steps: int):
    """Autoregressively predict n_steps frames from one frame.

    x: (B, 1, Lx, Ly, F), as WellDataset serves input_fields.
    cond: (B, 2), as `conditioning` returns it.
    returns: (B, n_steps, Lx, Ly, F), same layout as output_fields.
    """
    m0 = rearrange(x, "B 1 Lx Ly F -> B F Lx Ly")

    def step(m, _):
        m = jax.vmap(model)(m, cond)
        return m, m

    _, frames = jax.lax.scan(step, m0, None, length=n_steps)
    return rearrange(frames, "T B F Lx Ly -> B T Lx Ly F")


def one_step_preds(model, loader):
    """Teacher-forced one-step predictions over a whole loader.

    returns: (pred, truth), both (N, 1, Lx, Ly, F) numpy -- the layout the
    well's metrics reduce over.
    """
    preds, truths = [], []
    for batch in tqdm(loader, mininterval=10):
        m0, m1, cond = prepare_batch(batch)
        preds.append(np.asarray(predict(model, m0, cond)))
        truths.append(m1)

    pattern = "B F Lx Ly -> B 1 Lx Ly F"
    return (
        rearrange(np.concatenate(preds), pattern),
        rearrange(np.concatenate(truths), pattern),
    )


def relative_norm_error(m):
    norm = np.linalg.norm(m, axis=-1)
    return norm.mean((1, 2))


def device_info():
    print(jax.devices())
