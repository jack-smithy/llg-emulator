"""Where does the SP4 pointwise error come from? Four measurements.

(a) teacher-forced one-step error   -> how good the learned map itself is
(b) free-running rollout error      -> what the 100-step unroll accumulates
    (a) tiny while (b) large == compounding, not capacity.
(c) best time-shift per frame       -> is the unroll just running early/late?
(d) error vs |grad m| of reference  -> is it localised on domain walls?
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from llg_emulator.config import JAX_CACHE_DIR, sp4_path
from llg_emulator.data import _frames_to_cf, open_trajectory
from llg_emulator.io import load_model
from llg_emulator.model import with_mesh
from llg_emulator.rollout import rollout_trajectory

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)

RUN = __import__("sys").argv[1]
trj = open_trajectory(sp4_path())
m_ref = jnp.asarray(_frames_to_cf(trj.m))                      # (t, 3, h, w)
model = load_model(Path("" + RUN + ""), key=jr.PRNGKey(0), tag="best")
model = with_mesh(model, trj.n, trj.dx)
cond = jnp.concatenate((jnp.asarray(trj.H), jnp.array([trj.s_enc(1)])))

# (a) teacher-forced: one step from every *true* frame
tf_pred = jax.vmap(lambda m: model(m, cond))(m_ref[:-1])
tf_err = np.asarray(jnp.linalg.norm(tf_pred - m_ref[1:], axis=1))   # (t-1, h, w)

# (b) free-running
fr_pred = rollout_trajectory(model, m_ref, jnp.asarray(trj.H), s0=jnp.array([trj.s_enc(1)]))
fr_err = np.asarray(jnp.linalg.norm(fr_pred - m_ref, axis=1))       # (t, h, w)

# (c) does a small time shift explain the free-running error?
ref = np.asarray(m_ref)
pred = np.asarray(fr_pred)
shifts, shifted = [], []
for t in range(len(pred)):
    lo, hi = max(0, t - 8), min(len(ref), t + 9)
    d = [np.linalg.norm(pred[t] - ref[k], axis=0).mean() for k in range(lo, hi)]
    k = int(np.argmin(d))
    shifts.append(lo + k - t)
    shifted.append(d[k])
shifts, shifted = np.array(shifts), np.array(shifted)

# (d) is the error where the reference varies fastest in space?
gx = np.diff(ref, axis=2, prepend=ref[:, :, :1])
gy = np.diff(ref, axis=3, prepend=ref[:, :, :, :1])
grad = np.linalg.norm(gx, axis=1) + np.linalg.norm(gy, axis=1)      # (t, h, w)
mask = grad[1:] > np.percentile(grad[1:], 90)
out = {
    "tf_mean": tf_err.mean(axis=(1, 2)).tolist(),
    "tf_max": tf_err.max(axis=(1, 2)).tolist(),
    "fr_mean": fr_err.mean(axis=(1, 2)).tolist(),
    "fr_max": fr_err.max(axis=(1, 2)).tolist(),
    "shift": shifts.tolist(),
    "shifted_mean": shifted.tolist(),
    "raw_mean": [float(np.linalg.norm(pred[t] - ref[t], axis=0).mean()) for t in range(len(pred))],
}
print(f"(a) teacher-forced one-step: mean {tf_err.mean():.5f}  max {tf_err.max():.5f}")
print(f"(b) free-running rollout:    mean {fr_err.mean():.5f}  max {fr_err.max():.5f}")
print(f"    ratio of means (b)/(a):  {fr_err.mean() / tf_err.mean():.1f}x")
print(f"(c) best time shift: median {np.median(shifts):+.0f} frames, "
      f"mean |e| at best shift {shifted.mean():.4f} vs {np.mean(out['raw_mean']):.4f} unshifted "
      f"({100 * (1 - shifted.mean() / np.mean(out['raw_mean'])):.0f}% of the error is timing)")
print(f"(d) one-step error on the top-10% |grad m| nodes: {tf_err[mask].mean():.5f} "
      f"vs {tf_err[~mask].mean():.5f} elsewhere "
      f"({tf_err[mask].mean() / tf_err[~mask].mean():.1f}x)")
Path(__file__).with_name(f"diagnose_{RUN.split(chr(47))[-1]}.json").write_text(json.dumps(out))
