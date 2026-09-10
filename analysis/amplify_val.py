"""Amplification (rollout error / one-step error) on in-distribution val trajectories.

The SP4 diagnostic only covers the out-of-distribution field. Pushforward is meant
to fix compounding on the model's *own* error distribution, which is what dominates
in-distribution rollouts -- so measure it there too.
"""
import json, sys
from pathlib import Path

import jax, jax.numpy as jnp, jax.random as jr, numpy as np

from llg_emulator.config import BENCHMARK_VARIANTS, JAX_CACHE_DIR, variant_dirs
from llg_emulator.data import _frames_to_cf, open_trajectory, sample_dirs
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh
from llg_emulator.rollout import rollout_trajectory

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
N = 6
out = {}
paths = sample_dirs(variant_dirs("val", exclude=BENCHMARK_VARIANTS))[:N]
for run in sys.argv[1:]:
    model = load_model(Path(run), key=jr.PRNGKey(0), tag="best")
    tf_all, fr_all = [], []
    for p in paths:
        trj = open_trajectory(p)
        m = with_mesh(model, trj.n, trj.dx)
        ref = jnp.asarray(_frames_to_cf(trj.m))
        cond = jnp.concatenate((jnp.asarray(trj.H), jnp.array([trj.s_enc(1)])))
        tf = jax.vmap(lambda x: m(x, cond))(ref[:-1])
        tf_all.append(float(jnp.linalg.norm(tf - ref[1:], axis=1).mean()))
        fr = rollout_trajectory(m, ref, jnp.asarray(trj.H), s0=jnp.array([trj.s_enc(1)]))
        fr_all.append(float(jnp.linalg.norm(fr - ref, axis=1).mean()))
    tf_m, fr_m = float(np.mean(tf_all)), float(np.mean(fr_all))
    out[run] = {"tf": tf_m, "fr": fr_m, "ratio": fr_m / tf_m}
    print(f"{run:18s} val one-step {tf_m:.5f}  rollout {fr_m:.5f}  amplification {fr_m/tf_m:5.1f}x")
Path(__file__).with_name("amplify_val.json").write_text(json.dumps(out))
