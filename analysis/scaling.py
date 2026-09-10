"""Seconds per 10 ps step vs number of nodes, learned integrator and dopri5 reference,
same GPU, JIT-warm. Only meshes whose demag tensor is already in .nm_cache (no builds).
State for both: S-state under the SP4 field, so the adaptive solver sees comparable
dynamics at every size (its substep count is set by exchange stiffness at fixed dx)."""
import json, time
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx, neuralmag as nm
from pathlib import Path
from llg_emulator.config import JAX_CACHE_DIR
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh
from datagen.generate import NM_CACHE, init_m_s_state
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
nm.config.dtype = "float32"; nm.set_log_level(30)
DX = (5e-9, 5e-9, 3e-9); H = [-19576.058, 3421.83, 0.0]; DT = 1e-11
MESHES = [(16, 16), (31, 31), (63, 63), (100, 25), (256, 256), (2000, 2000), (4000, 4000), (5000, 5000)]

def sstate(n):
    m = np.zeros((n[0] + 1, n[1] + 1, 3), np.float32); m[1:-1, :, 0] = 1; m[(0, -1), :, 1] = 1
    return m / np.linalg.norm(m, axis=-1, keepdims=True)

def solver_step_time(n, reps):
    st = nm.State(nm.Mesh(n, DX)); st.material.Ms, st.material.A, st.material.alpha = 8e5, 1.3e-11, 0.02
    st.m = nm.VectorFunction(st, tensor=st.tensor(sstate(n)))
    nm.ExchangeField().register(st, "exchange"); nm.DemagField(cache_dir=NM_CACHE).register(st, "demag")
    nm.ExternalField(st.tensor(H)).register(st, "external"); nm.TotalField("exchange", "demag", "external").register(st)
    llg = nm.LLGSolver(st)
    for _ in range(2): llg.step(DT)
    np.asarray(st.m.tensor); t0 = time.perf_counter()
    for _ in range(reps): llg.step(DT)
    np.asarray(st.m.tensor); return (time.perf_counter() - t0) / reps

def model_step_time(model, n, reps):
    m = with_mesh(model, n, DX); cond = jnp.concatenate((jnp.asarray(H, jnp.float32) / 8e5, jnp.zeros(1)))
    step = eqx.filter_jit(lambda x, c: m(x, c)); x = jnp.asarray(np.moveaxis(sstate(n), -1, 0))
    for _ in range(2): x = step(x, cond)
    x.block_until_ready(); t0 = time.perf_counter()
    for _ in range(reps): x = step(x, cond)
    x.block_until_ready(); return (time.perf_counter() - t0) / reps

model = eqx.nn.inference_mode(load_model(Path("weights/run1"), key=jr.PRNGKey(0), tag="best"))
out = []
print(f"{'mesh':>12} {'nodes':>10} {'solver s/step':>14} {'model s/step':>13} {'ratio':>6}")
for n in MESHES:
    nodes = (n[0] + 1) * (n[1] + 1); reps = 20 if nodes < 5e6 else 10
    ts = solver_step_time(n, reps)
    try: tm = model_step_time(model, n, reps)
    except Exception as e: tm = float("nan"); print(f"   model OOM at {n}: {type(e).__name__}")
    out.append(dict(n=n, nodes=nodes, solver=ts, model=tm))
    print(f"{str(n):>12} {nodes:10d} {ts:14.4f} {tm:13.4f} {ts/tm if tm == tm else float('nan'):6.1f}", flush=True)
Path(__file__).with_name("scaling.json").write_text(json.dumps(out))
