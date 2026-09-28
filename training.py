"""Training core: the loss, the jitted update step, and checkpointing.

Ported from `src/llg_emulator/{training,io}.py` on main. The objective is a
plain **one-step MSE** on `(m_t, m_{t+1})` pairs. The demag tensor is held out
of the gradient by `trainable_filter`, which is also what `save_model` filters
on so the frozen tensor never reaches a checkpoint.
"""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree

from resnet import LLGEmulator, ModelConfig
from physics import demag_cache


def trainable_filter(model: LLGEmulator):
    spec = jtu.tree_map(eqx.is_inexact_array, model)
    return eqx.tree_at(lambda m: m.demag.N, spec, replace=False)


def count_parameters(model: LLGEmulator) -> int:
    return sum(
        p.size for p in jtu.tree_leaves(eqx.filter(model, trainable_filter(model)))
    )


@eqx.filter_jit
def loss_fn(model, m0, m1, cond):
    return jnp.mean(jnp.square(jax.vmap(model)(m0, cond) - m1))


# The model is not donated: `demag.N` rides along in it as a frozen array leaf,
# and donating it invalidates the `demag_cache`d operator the model was
# built from, so a later `load_model`/`with_mesh` on the same mesh would hit a
# deleted buffer. The params it costs a copy of are a few MB.
@eqx.filter_jit(donate="all-except-first")
def update_fn(model: LLGEmulator, batch: PyTree, optimizer, opt_state):
    diff, static = eqx.partition(model, trainable_filter(model))

    def diff_loss(diff):
        return loss_fn(eqx.combine(diff, static), *batch)

    loss, grad = eqx.filter_value_and_grad(diff_loss)(diff)
    updates, opt_state = optimizer.update(grad, opt_state, params=diff)
    diff = eqx.apply_updates(diff, updates)
    return eqx.combine(diff, static), opt_state, loss


def save_model(model: LLGEmulator, config: ModelConfig, path: Path, tag: str):
    """Serialise the *trainable* leaves to `<path>/<tag>.eqx` + `metadata.json`.

    The frozen demag tensor is most of the pytree and `load_model` rebuilds it
    exactly from the sidecar mesh, so writing it out is pure bloat.
    """
    path.mkdir(parents=True, exist_ok=True)
    params, _ = eqx.partition(model, trainable_filter(model))
    eqx.tree_serialise_leaves(path / f"{tag}.eqx", params)
    meta = {
        "hidden_channels": config.hidden_channels,
        "num_blocks": config.num_blocks,
        "cond_dim": config.cond_dim,
        # the mesh the weights were trained on: enough to rebuild the demag
        # operator without the caller having to know anything about the run
        "mesh_n": list(model.demag.n),
        "mesh_dx": list(model.demag.dx),
        "demag_p": model.demag.p,
        "nodal": model.demag.nodal,
    }
    # rewrite every time: a stale sidecar builds the wrong skeleton on load
    (path / "metadata.json").write_text(json.dumps(meta))


def load_model(path: Path, key, tag: str, n=None) -> LLGEmulator:
    """Rebuild the model from `<tag>.eqx` + `metadata.json`.

    Pass `n` (mesh cell counts) to retarget the model at a different grid than
    it was trained on -- the backbone is fully convolutional, only the demag
    tensor is mesh-shaped.
    """
    meta = json.loads((path / "metadata.json").read_text())
    config = ModelConfig(
        hidden_channels=meta["hidden_channels"],
        num_blocks=meta["num_blocks"],
        cond_dim=meta["cond_dim"],
        mesh_n=tuple(meta["mesh_n"]),
        mesh_dx=tuple(meta["mesh_dx"]),
        demag_p=meta["demag_p"],
        nodal=meta["nodal"],
    )
    demag = demag_cache(
        tuple(n) if n is not None else config.mesh_n,
        config.mesh_dx,
        1.0,
        config.demag_p,
        nodal=config.nodal,
    )
    skeleton = LLGEmulator(config=config, demag=demag, key=key)
    params, static = eqx.partition(skeleton, trainable_filter(skeleton))
    params = eqx.tree_deserialise_leaves(path / f"{tag}.eqx", params)
    return eqx.combine(params, static)
