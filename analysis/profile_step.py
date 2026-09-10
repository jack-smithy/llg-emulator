"""Where does a learned-integrator step spend its time at 4001^2? Isolates: JIT compile,
demag FFT, CNN, tangent update, and the eval harness (memmap read + CPU transpose + H2D)."""
import time, sys
import numpy as np, jax, jax.numpy as jnp, jax.random as jr, equinox as eqx
from pathlib import Path
from llg_emulator.config import JAX_CACHE_DIR, benchmark_path
from llg_emulator.data import _frame_to_cf, open_trajectory
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)

variant = sys.argv[1] if len(sys.argv) > 1 else "sp4_xlarge"
trj = open_trajectory(benchmark_path(variant))
model = with_mesh(eqx.nn.inference_mode(load_model(Path("weights/run1"), key=jr.PRNGKey(0), tag="best")), trj.n, trj.dx)
cond = jnp.concatenate((jnp.asarray(trj.H), jnp.array([trj.s_enc(1)])))
x = jnp.asarray(_frame_to_cf(trj.m, 0)); x.block_until_ready()
print(f"mesh {trj.n}, nodal {x.shape[1:]}, {x.size*4/1e6:.0f} MB/frame, device {jax.devices()[0].device_kind}")

def timeit(fn, *a, n=10, label=""):
    fn(*a).block_until_ready()                       # warm (compile excluded)
    t0 = time.perf_counter()
    for _ in range(n): y = fn(*a)
    y.block_until_ready()
    dt = (time.perf_counter() - t0) / n
    print(f"  {label:<44s} {dt*1e3:8.1f} ms", flush=True); return dt

# --- compile cost of the full step
step = eqx.filter_jit(lambda m: model(m, cond))
t0 = time.perf_counter(); step(x).block_until_ready(); t_compile = time.perf_counter() - t0
print(f"  {'full step: compile + first call':<44s} {t_compile*1e3:8.1f} ms")

# --- device-side components, jit-warm
t_full  = timeit(step, x, label="full step (demag + CNN + update), warm")
t_demag = timeit(eqx.filter_jit(model.demag), x, label="  demag: to_cell -> rfft2 -> N -> irfft2 -> to_node")
six = jnp.concatenate([x, model.demag(x)], 0)
def cnn_only(inp):
    mod = model.init_film(cond)
    h = mod(model.backbone.lifting(inp), 0)
    for i, blk in enumerate(model.backbone.blocks, 1): h = mod(blk(h), i)
    return model.backbone.projection(h)
t_cnn   = timeit(eqx.filter_jit(cnn_only), six, label="  CNN: lifting + 4 res blocks + projection")
dm = cnn_only(six)
t_upd   = timeit(eqx.filter_jit(model.step), x, dm, label="  tangent projection + renormalise")
print(f"  {'  sum of parts':<44s} {(t_demag+t_cnn+t_upd)*1e3:8.1f} ms   (vs full {t_full*1e3:.1f})")

# --- eval-harness overhead per frame, as rollout_metrics does it
t0 = time.perf_counter()
for t in range(1, 11): f = _frame_to_cf(trj.m, t)                       # memmap read + CPU transpose
t_cpu = (time.perf_counter() - t0) / 10
t0 = time.perf_counter()
for t in range(11, 21): r = jnp.asarray(_frame_to_cf(trj.m, t)); r.block_until_ready()
t_h2d = (time.perf_counter() - t0) / 10
r = jnp.asarray(_frame_to_cf(trj.m, 1)); m = step(x)
t0 = time.perf_counter()
for _ in range(10):
    a = np.asarray(r.mean(axis=(1, 2))); b = np.asarray(m.mean(axis=(1, 2))); c = float(jnp.mean(jnp.sum(m*r, 0)))
t_red = (time.perf_counter() - t0) / 10
print(f"  {'harness: memmap read + CPU transpose / frame':<44s} {t_cpu*1e3:8.1f} ms")
print(f"  {'harness: ... + host->device transfer':<44s} {t_h2d*1e3:8.1f} ms")
print(f"  {'harness: 3 reductions w/ device sync':<44s} {t_red*1e3:8.1f} ms")
per = t_full + t_h2d + t_red
print(f"\n  per-step total as rollout_metrics runs it ~ {per*1e3:.0f} ms  -> 100 steps ~ {per*100:.1f} s (+{t_compile:.1f} s compile)")
print(f"  model alone: {t_full*1e3:.0f} ms/step -> 100 steps {t_full*100:.1f} s;  demag share of model {100*t_demag/t_full:.0f}%, CNN {100*t_cnn/t_full:.0f}%")
