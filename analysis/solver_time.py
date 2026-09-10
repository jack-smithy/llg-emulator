"""Wall time for neuralmag to produce the 1 ns trajectory (100 x 10 ps) of each
benchmark, starting from the stored frame 0 -- no relax, so it is exactly the
work the surrogate replaces."""
import json, time
import numpy as np, neuralmag as nm
from llg_emulator.config import BENCHMARK_VARIANTS, benchmark_path
from datagen.generate import NM_CACHE
nm.config.dtype = "float32"; nm.set_log_level(30)

for variant in BENCHMARK_VARIANTS:
    p = benchmark_path(variant); md = json.load(open(p / "metadata.json"))
    m = np.load(p / "m.npy", mmap_mode="r")
    st = nm.State(nm.Mesh(tuple(md["n"]), tuple(md["dx"])))
    for k, v in md["material"].items(): setattr(st.material, k, v)
    st.m = nm.VectorFunction(st, tensor=st.tensor(np.asarray(m[0])))
    nm.ExchangeField().register(st, "exchange")
    nm.DemagField(cache_dir=NM_CACHE).register(st, "demag")
    nm.ExternalField(st.tensor(md["H_ext"])).register(st, "external")
    nm.TotalField("exchange", "demag", "external").register(st)
    llg = nm.LLGSolver(st)
    llg.step(md["dt"]); np.asarray(st.m.tensor)          # warm-up / compile
    t0 = time.perf_counter()
    for _ in range(100): llg.step(md["dt"])
    np.asarray(st.m.tensor)
    secs = time.perf_counter() - t0
    err = float(np.linalg.norm(np.asarray(st.m.tensor) - m[100], axis=-1).mean())
    print(f"{variant:>10} mesh {tuple(md['n'])!s:>12}: 100 steps {secs:7.1f}s  ({secs/100:.3f} s/step)"
          f"  |m-ref_100| {err:.2e} (re-run vs stored: solver self-consistency)", flush=True)
