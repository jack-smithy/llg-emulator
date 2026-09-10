"""Per-stride one-step MSE on the val films: is the mixed-stride val number bad
everywhere, or only where the step is long? Trivial baseline per stride for scale."""
import json, sys
import numpy as np, jax, jax.numpy as jnp, jax.random as jr
from pathlib import Path
from llg_emulator.config import BENCHMARK_VARIANTS, JAX_CACHE_DIR, variant_dirs
from llg_emulator.data import LLGStepperSource
from llg_emulator.io import load_model
from llg_emulator.training import loss_fn
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)

STRIDES = (1, 2, 4, 8, 16)
src = LLGStepperSource(variant_dirs("val", exclude=BENCHMARK_VARIANTS), strides=STRIDES, max_trajectories=16)
rng = np.random.default_rng(0)
out = {}
models = {r: load_model(Path(r), key=jr.PRNGKey(0), tag="best") for r in sys.argv[1:]}
print(f"{'stride':>6} {'trivial':>10} " + " ".join(f"{r.split('/')[-1]:>10}" for r in models))
for s in STRIDES:
    idx = rng.choice(np.where(src.index[:, 2] == s)[0], 160, replace=False)
    triv, res = [], {r: [] for r in models}
    for chunk in np.array_split(idx, 4):
        b = {k: jnp.asarray(np.stack([src[int(i)][k] for i in chunk])) for k in ("m0", "m1", "m2", "H", "s0")}
        triv.append(float(jnp.mean((b["m1"] - b["m0"]) ** 2)))
        for r, m in models.items():
            res[r].append(float(loss_fn(m, **b)))
    out[s] = {"trivial": float(np.mean(triv)), **{r: float(np.mean(v)) for r, v in res.items()}}
    print(f"{s:6d} {out[s]['trivial']:10.3e} " + " ".join(f"{out[s][r]:10.3e}" for r in models), flush=True)
Path(__file__).with_name("val_by_stride.json").write_text(json.dumps(out))
