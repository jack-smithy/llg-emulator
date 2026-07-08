"""Training core: losses, the jitted update step, dataloading source, LR
schedule, and model (de)serialisation.

The training objective is a **k-step rollout MSE**: the model is unrolled on its
own predictions for `k` steps and every intermediate frame is matched, so the
gradient sees the compounding error that one-step training ignores (k=1 recovers
plain one-step MSE). Best-checkpointing selects on the SP4 rollout (`sp4_rollout_rmse`).
"""

import json
from pathlib import Path

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
from jaxtyping import Array, PyTree

from llg_emulator.config import dataset_dir
from llg_emulator.data import LLGStepperSource, load_trajectories, load_trajectory
from llg_emulator.metrics import MSE
from llg_emulator.model import LLGEmulator, ModelConfig
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.symmetry import NUM_ELEMENTS, apply_field, apply_H


def trainable_filter(model: LLGEmulator):
    """Bool pytree: True for trainable leaves, False for the fixed demag tensor.

    The demag submodule is a physical operator with no learnable parameters,
    but its precomputed tensor `demag.N` is a large array leaf that the default
    is_array filter would otherwise feed to the optimiser. Freeze it.
    """
    spec = jtu.tree_map(eqx.is_inexact_array, model)
    return eqx.tree_at(lambda m: m.demag.N, spec, replace=False)


def count_parameters(model: LLGEmulator) -> int:
    return sum(
        p.size for p in jtu.tree_leaves(eqx.filter(model, trainable_filter(model)))
    )


# --- rollout-k training objective --------------------------------------------


def rollout_loss_fn(model: LLGEmulator, m_seq: Array, H: Array, k: int) -> Array:
    """k-step unrolled MSE for one trajectory window.

    m_seq: (k+1, 3, nx, ny) consecutive ground-truth frames. The model is fed its
    own prediction for k steps and every intermediate frame is matched, so the
    gradient sees compounding error. k=1 == one-step MSE.
    """
    def scan_fn(m, m_target):
        m_next = model(m, H)
        return m_next, MSE(m_next, m_target)

    _, losses = jax.lax.scan(scan_fn, m_seq[0], m_seq[1:], length=k)
    return jnp.mean(losses)


def loss_fn(model: LLGEmulator, m_seq: Array, H: Array, k: int) -> Array:
    """Batched rollout loss. m_seq (b, k+1, 3, nx, ny), H (b, 3)."""
    return jnp.mean(jax.vmap(lambda s, h: rollout_loss_fn(model, s, h, k))(m_seq, H))


@eqx.filter_jit(donate="all")
def update_fn(model: LLGEmulator, batch: PyTree, optimizer, opt_state, k: int):
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), batch["m_seq"], batch["H"], k)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state, params=diff)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


def train_epoch(model, loader: grain.IterDataset, optimizer, opt_state, k: int):
    losses = []
    for batch in loader:
        model, opt_state, loss = update_fn(model, batch, optimizer, opt_state, k)
        losses.append(loss)
    return model, opt_state, jnp.stack(losses).mean().item()


# --- one-step val loss (held-out) --------------------------------------------


@eqx.filter_jit(donate="all-except-first")
def _val_step(model: LLGEmulator, m0: Array, m1: Array, H: Array):
    return MSE(jax.vmap(model)(m0, H), m1)


def val_epoch(model, loader: grain.IterDataset) -> float:
    """One-step MSE on the val split (loader yields (m0, m1, H) batches)."""
    inference_model = eqx.nn.inference_mode(model)
    losses = []
    for batch in loader:
        losses.append(_val_step(inference_model, batch["m0"], batch["m1"], batch["H"]))
    return jnp.stack(losses).mean().item()


# --- rollout dataloading source ----------------------------------------------


class RolloutSource(LLGStepperSource):
    """Length-(k+1) rollout windows, optionally expanded by the D4xZ2 symmetry.

    k=1 reduces to consecutive (m0, m1) pairs. With augment=True the index is
    multiplied by the 16 exact symmetry elements (see symmetry.py): each window
    is transformed on access, giving 16x more physically-exact samples spanning
    all field quadrants — physically exact, verified demag-equivariant.
    """

    def __init__(self, path, k: int, augment: bool = False, max_workers: int = 16):
        self.trajs, self.fields = load_trajectories(path, max_workers)
        self.k = k
        self.n_groups = NUM_ELEMENTS if augment else 1
        self.index = np.array(
            [
                (ti, t, g)
                for ti, trj in enumerate(self.trajs)
                for t in range(trj.shape[0] - k)
                for g in range(self.n_groups)
            ],
            dtype=np.int64,
        )

    def __getitem__(self, idx):
        ti, t, g = self.index[idx]
        m_seq = self.trajs[ti][t : t + self.k + 1]
        H = self.fields[ti]
        if g != 0:
            m_seq = apply_field(m_seq, int(g))  # broadcasts over the time axis
            H = apply_H(H, int(g))
        return {"m_seq": m_seq, "H": H}


# --- LR schedule -------------------------------------------------------------


def make_schedule(lr, epochs, steps_per_epoch, warmup_frac=0.05):
    """Warmup + cosine decay over the whole run."""
    total = epochs * steps_per_epoch
    warmup = max(1, int(total * warmup_frac))
    return optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=warmup,
        decay_steps=total,
        end_value=lr * 0.02,
    )


# --- (de)serialisation + SP4 checkpoint metric -------------------------------


def save_model(model: LLGEmulator, config: ModelConfig, path):
    """Serialise weights + a sidecar json holding the arch needed to rebuild."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    meta = {"hidden_channels": config.hidden_channels, "num_blocks": config.num_blocks}
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_model(path) -> LLGEmulator:
    """Rebuild a model from weights + its .json sidecar."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    config = ModelConfig(
        hidden_channels=meta["hidden_channels"], num_blocks=meta["num_blocks"]
    )
    skeleton = LLGEmulator(config=config, key=jr.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, skeleton)


def sp4_rollout_rmse(model: LLGEmulator, sp4_path: Path) -> float:
    """Bulk-magnetization RMSE of a 100-step SP4 rollout vs. the reference.

    The model-selection metric: SP4 is the deliverable and its applied field is
    out-of-distribution, so this is what best-checkpointing tracks.
    """
    model = eqx.nn.inference_mode(model)
    m_true, H = load_trajectory(sp4_path)
    m_true = jnp.asarray(m_true)
    m_pred = rollout_trajectory(model, m_true, jnp.asarray(H), include_init=True)
    bulk_true = jnp.mean(m_true, axis=(2, 3))
    bulk_pred = jnp.mean(m_pred, axis=(2, 3))
    return float(jnp.sqrt(jnp.mean((bulk_true - bulk_pred) ** 2)))
