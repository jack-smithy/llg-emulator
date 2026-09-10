"""Solver limit (cached 5000 demag) and model limit at 4000."""
import time, sys
import jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, numpy as np
from pathlib import Path

def gb(): return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9

print("device:", jax.devices()[0].device_kind,
      f"{jax.devices()[0].memory_stats()['bytes_limit']/1e9:.1f} GB")

# --- solver at 5000^2 (demag cached): relax + 2 steps
import neuralmag as nm
nm.config.dtype = "float32"; nm.set_log_level(30)
from datagen.generate import NM_CACHE, init_m_s_state
for n in (5000,):
    t0 = time.time()
    mesh = nm.Mesh((n, n), (5e-9, 5e-9, 3e-9)); st = nm.State(mesh)
    st.material.Ms, st.material.A, st.material.alpha = 8e5, 1.3e-11, 0.02
    init_m_s_state(st, inv=False)
    nm.ExchangeField().register(st, "exchange")
    nm.DemagField(cache_dir=NM_CACHE).register(st, "demag")
    nm.TotalField("exchange", "demag").register(st)
    llg = nm.LLGSolver(st); llg.relax()
    t1 = time.time()
    nm.ExternalField(st.tensor([-24.6e-3/(4e-7*np.pi), 4.3e-3/(4e-7*np.pi), 0.0])).register(st, "external")
    nm.TotalField("exchange", "demag", "external").register(st); llg.reset(); st.t = 0.0
    for _ in range(2): llg.step(10e-12)
    np.asarray(st.m.tensor)
    print(f"SOLVER n={n}: OK  relax {t1-t0:.0f}s  2 steps {time.time()-t1:.0f}s  peak {gb():.1f} GB")

# --- model at 4000^2 (builds + caches the demag tensor)
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh
model = load_model(Path("weights/run1"), key=jr.PRNGKey(0), tag="best")
for n in (4000,):
    try:
        t0 = time.time()
        m = with_mesh(eqx.nn.inference_mode(model), (n, n), (5e-9, 5e-9, 3e-9))
        x = jnp.zeros((3, n + 1, n + 1), jnp.float32).at[2].set(1.0)
        step = eqx.filter_jit(lambda a: m(a, jnp.zeros(4)))
        y = step(x); y.block_until_ready()
        t1 = time.time()
        for _ in range(3): y = step(y)
        y.block_until_ready()
        print(f"MODEL  n={n}: OK  build+compile {t1-t0:.0f}s  peak {gb():.1f} GB  "
              f"|m|={float(jnp.linalg.norm(y,axis=0).mean()):.4f}")
    except Exception as e:
        print(f"MODEL  n={n}: FAILED {type(e).__name__}: {str(e)[:100]}")
