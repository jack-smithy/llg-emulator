import json
from pathlib import Path

import equinox as eqx
import numpy as np
from pyevtk.hl import gridToVTK

from llg_emulator.model import LLGEmulator, ModelConfig
from llg_emulator.physics import demag_for
from llg_emulator.training import trainable_filter


def write_vtr(field, filename, n, dx):
    # save result
    x = np.arange(0, (0.1 + n[0]) * dx[0], dx[0], dtype="float64")
    y = np.arange(0, (0.1 + n[1]) * dx[1], dx[1], dtype="float64")
    z = np.arange(0, (0.1 + n[2]) * dx[2], dx[2], dtype="float64")

    # scalar data
    if len(field.shape) == 3:
        gridToVTK(filename, x, y, z, cellData={"f": field.copy()})
    else:
        gridToVTK(
            filename,
            x,
            y,
            z,
            cellData={
                "f": (
                    field[:, :, :, 0].copy(),
                    field[:, :, :, 1].copy(),
                    field[:, :, :, 2].copy(),
                )
            },
        )


def save_model(model: LLGEmulator, config: ModelConfig, path: Path, tag: str):
    """Serialise the *trainable* leaves only.

    The frozen demag tensor is ~78% of the pytree (4.7 MB of a 6.1 MB 256x256
    checkpoint) and `load_model` rebuilds it exactly from the sidecar mesh, so
    writing it out is pure bloat. Checkpoints written before this change hold the
    full tree and will not load against the filtered skeleton.
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
    }
    # rewrite every time: a stale sidecar builds the wrong skeleton on load
    (path / "metadata.json").write_text(json.dumps(meta))


def load_model(path: Path, key, tag: str, n=None) -> LLGEmulator:
    """Rebuild the model from `<tag>.eqx` + `metadata.json`.

    The demag operator is rebuilt from the sidecar rather than passed in, so a
    checkpoint is self-describing. Pass `n` (mesh *cell* counts) to retarget the
    model at a different grid than it was trained on — the backbone is fully
    convolutional, only the demag tensor is mesh-shaped.
    """
    meta = json.loads((path / "metadata.json").read_text())
    config = ModelConfig(
        hidden_channels=meta["hidden_channels"],
        num_blocks=meta["num_blocks"],
        cond_dim=meta["cond_dim"],
        mesh_n=tuple(meta["mesh_n"]),
        mesh_dx=tuple(meta["mesh_dx"]),
        demag_p=meta["demag_p"],
    )
    demag = demag_for(
        tuple(n) if n is not None else config.mesh_n,
        config.mesh_dx,
        1.0,
        config.demag_p,
    )
    skeleton = LLGEmulator(config=config, demag=demag, key=key)
    params, static = eqx.partition(skeleton, trainable_filter(skeleton))
    params = eqx.tree_deserialise_leaves(path / f"{tag}.eqx", params)
    return eqx.combine(params, static)


def _demo():
    """Self-check: round-trip the weights, drop the demag tensor, retarget the mesh."""
    import tempfile

    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu

    config = ModelConfig(hidden_channels=16, num_blocks=2, mesh_n=(15, 15))
    demag = demag_for(config.mesh_n, config.mesh_dx, 1.0, config.demag_p)
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    model = LLGEmulator(config=config, demag=demag, key=k1)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d)
        save_model(model, config, path, tag="weights")
        # k2 != k1, so a leaf that is not actually restored shows up as a mismatch
        loaded = load_model(path, key=k2, tag="weights")
        smaller = load_model(path, key=k2, tag="weights", n=(7, 7))

        nbytes = (path / "weights.eqx").stat().st_size
        full = sum(
            x.size * x.dtype.itemsize
            for x in jtu.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
        )
        assert nbytes < full, f"no saving: {nbytes} >= {full}"

    for a, b in zip(
        jtu.tree_leaves(eqx.filter(model, trainable_filter(model))),
        jtu.tree_leaves(eqx.filter(loaded, trainable_filter(loaded))),
    ):
        assert jnp.array_equal(a, b), "trainable leaf not round-tripped"
    assert jnp.array_equal(loaded.demag.N, demag.N), "demag not rebuilt from mesh"

    m = jax.random.normal(k1, (3, 16, 16))
    m = m / jnp.linalg.norm(m, axis=0, keepdims=True)
    cond = jnp.zeros(config.cond_dim)
    assert jnp.allclose(model(m, cond), loaded(m, cond)), "outputs differ after reload"

    # same weights, different grid: the whole point of the mesh metadata
    assert smaller.demag.n == (7, 7)
    small = jax.random.normal(k2, (3, 8, 8))
    small = small / jnp.linalg.norm(small, axis=0, keepdims=True)
    out = smaller(small, cond)
    assert out.shape == (3, 8, 8), out.shape
    assert jnp.allclose(jnp.linalg.norm(out, axis=0), 1.0, atol=1e-5)
    print(
        f"io self-check ok: {nbytes / 1e6:.2f} MB written vs {full / 1e6:.2f} MB full; "
        f"reload exact; retargeted 15x15 -> 7x7"
    )


if __name__ == "__main__":
    _demo()
