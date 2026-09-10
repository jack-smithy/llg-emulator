"""How reproducible is the reference solver itself? Two dopri5 re-runs from the stored
frame 0 -- one exact, one with a 1e-6 random perturbation -- scored against the stored
trajectory with the same metrics used for the learned integrator (per-frame pointwise
error, bulk <m>(t) RMSE, final cosine), plus the divergence between the two re-runs."""
import json
import numpy as np, neuralmag as nm
from pathlib import Path
from llg_emulator.config import BENCHMARK_VARIANTS, benchmark_path
from datagen.generate import NM_CACHE
nm.config.dtype = "float32"; nm.set_log_level(30)

def rerun(md, m0):
    st = nm.State(nm.Mesh(tuple(md["n"]), tuple(md["dx"])))
    for k, v in md["material"].items(): setattr(st.material, k, v)
    st.m = nm.VectorFunction(st, tensor=st.tensor(np.asarray(m0, dtype=np.float32)))
    nm.ExchangeField().register(st, "exchange")
    nm.DemagField(cache_dir=NM_CACHE).register(st, "demag")
    nm.ExternalField(st.tensor(md["H_ext"])).register(st, "external")
    nm.TotalField("exchange", "demag", "external").register(st)
    llg = nm.LLGSolver(st); st.t = 0.0
    frames = [np.asarray(st.m.tensor)]
    for _ in range(100):
        llg.step(md["dt"]); frames.append(np.asarray(st.m.tensor))
    return frames

out = {}
for variant in BENCHMARK_VARIANTS:
    p = benchmark_path(variant); md = json.load(open(p / "metadata.json"))
    ref = np.load(p / "m.npy", mmap_mode="r")
    m0 = np.asarray(ref[0])
    pert = m0 + 1e-6 * np.random.default_rng(0).standard_normal(m0.shape).astype(np.float32)
    pert /= np.linalg.norm(pert, axis=-1, keepdims=True)
    A = rerun(md, m0); B = rerun(md, pert)
    err_A = [float(np.linalg.norm(a - ref[i], axis=-1).mean()) for i, a in enumerate(A)]
    err_AB = [float(np.linalg.norm(a - b, axis=-1).mean()) for a, b in zip(A, B)]
    bulk_ref = np.stack([np.asarray(ref[i]).mean(axis=(0, 1)) for i in range(101)])
    bulk_A = np.stack([a.mean(axis=(0, 1)) for a in A])
    rmse = float(np.sqrt(np.mean((bulk_A - bulk_ref) ** 2)))
    cos = float(np.mean(np.sum(A[-1] * ref[100], axis=-1)))
    out[variant] = {"err_vs_stored": err_A, "err_rerun_vs_rerun": err_AB, "bulk_rmse_vs_stored": rmse, "final_cos": cos}
    print(f"{variant:>10}: rerun vs stored  bulk_rmse {rmse:.4f}  pointwise mean {np.mean(err_A):.4f} final {err_A[-1]:.4f} cos {cos:.4f}"
          f"  |  rerun vs 1e-6-perturbed rerun: pointwise mean {np.mean(err_AB):.4f} final {err_AB[-1]:.4f}", flush=True)
Path(__file__).with_name("solver_repro.json").write_text(json.dumps(out))
