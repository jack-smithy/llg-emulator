import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
from pathlib import Path
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh
print(f"limit {jax.devices()[0].memory_stats()['bytes_limit']/1e9:.1f} GB")
model = load_model(Path("weights/run1"), key=jr.PRNGKey(0), tag="best")
for n in (5000,):
    try:
        m = with_mesh(eqx.nn.inference_mode(model), (n, n), (5e-9, 5e-9, 3e-9))
        x = jnp.zeros((3, n + 1, n + 1), jnp.float32).at[2].set(1.0)
        step = eqx.filter_jit(lambda a: m(a, jnp.zeros(4)))
        y = step(x); y.block_until_ready()
        for _ in range(3): y = step(y)
        y.block_until_ready()
        print(f"MODEL n={n}: OK peak {jax.devices()[0].memory_stats()['peak_bytes_in_use']/1e9:.1f} GB "
              f"|m|={float(jnp.linalg.norm(y,axis=0).mean()):.4f}")
    except Exception as e:
        print(f"MODEL n={n}: FAILED {str(e)[:110]}")
