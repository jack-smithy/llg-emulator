"""Streaming pointwise rollout error on data/val/large (2001^2 nodal, 4.85 GB)."""
import sys
from pathlib import Path
import equinox as eqx, jax.numpy as jnp, jax.random as jr, numpy as np, jax

from llg_emulator.config import JAX_CACHE_DIR, benchmark_path
from llg_emulator.data import _frame_to_cf, open_trajectory
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
trj = open_trajectory(benchmark_path(sys.argv[1]))
for run in sys.argv[2:]:
    model = with_mesh(eqx.nn.inference_mode(
        load_model(Path(run), key=jr.PRNGKey(0), tag="best")), trj.n, trj.dx)
    cond = jnp.concatenate((jnp.asarray(trj.H), jnp.array([trj.s_enc(1)])))
    step = eqx.filter_jit(lambda m: model(m, cond))
    m = jnp.asarray(_frame_to_cf(trj.m, 0)); errs = []
    for t in range(trj.m.shape[0]):
        if t: m = step(m)
        ref = jnp.asarray(_frame_to_cf(trj.m, t))
        errs.append(float(jnp.linalg.norm(m - ref, axis=0).mean()))
        if t == trj.m.shape[0] - 1:
            fin_max = float(jnp.linalg.norm(m - ref, axis=0).max())
    errs = np.array(errs)
    print(f"{run:16s} {sys.argv[1]:10s} rollout |e|: mean {errs.mean():.5f}  final {errs[-1]:.5f}  "
          f"final-frame max {fin_max:.5f}")
