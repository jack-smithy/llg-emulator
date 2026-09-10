"""Largest mesh the *model* can roll out on one GPU. Uses cached demag tensors only."""
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
from pathlib import Path
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh

d = jax.devices()[0]
print("device:", d.device_kind, f"{d.memory_stats()['bytes_limit']/1e9:.1f} GB limit")
model = load_model(Path("weights/run1"), key=jr.PRNGKey(0), tag="best")
for n in (2000, 3000, 5000):
    try:
        m = with_mesh(eqx.nn.inference_mode(model), (n, n), (5e-9, 5e-9, 3e-9))
        x = jnp.zeros((3, n + 1, n + 1), jnp.float32).at[2].set(1.0)
        step = eqx.filter_jit(lambda a: m(a, jnp.zeros(4)))
        y = step(x); y.block_until_ready()
        for _ in range(3):
            y = step(y)
        y.block_until_ready()
        peak = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
        print(f"  n={n:5d} nodal {n+1}^2  OK   peak {peak:.1f} GB  |m|={float(jnp.linalg.norm(y,axis=0).mean()):.4f}")
    except Exception as e:
        print(f"  n={n:5d}  FAILED: {type(e).__name__}: {str(e)[:120]}")
