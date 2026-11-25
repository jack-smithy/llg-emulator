"""Time per Barzilai-Borwein iteration against mesh size and precision, on this job's GPU:

- magnum.np's MinimizerBB in float64 (what datagen.py runs);
- train.relax with the true exchange (no learned term), float64 and float32.

Each runs N_ITERATIONS iterations (no convergence check) from a tilted uniform state at a reversing
field. Writes results/overnight/timing_<GPU>.json. Sizes that do not fit are recorded as "oom".
"""

import json
import time
from pathlib import Path

import torch
from magnumnp.common import Mesh, State
from magnumnp.field_terms import DemagField as MagnumDemag
from magnumnp.field_terms import ExchangeField as MagnumExchange
from magnumnp.field_terms import ExternalField
from magnumnp.solvers import MinimizerBB

import rollout
import train
from datagen import A, DX, Ms, material
from model import DemagField, ExchangeField

SIZES = [(100, 50, 20), (200, 100, 20), (400, 200, 20), (400, 200, 40), (800, 400, 40)]
N_ITERATIONS = 50
H_EXT = [-3e4, 5e3, 0.0]  # [A/m] about 38 mT against the start: the minimiser has to work


def start_state(n, dtype):
    m = torch.zeros((1, *n, 3), dtype=dtype, device="cuda")
    m[..., 0], m[..., 1] = 1.0, 0.1
    return torch.nn.functional.normalize(m + 0.05 * torch.randn_like(m), dim=-1)


def time_magnum(n):
    torch.set_default_dtype(torch.float64)
    state = State(Mesh(n=n, dx=DX))
    state.material.set(material)
    state.m = start_state(n, torch.float64)[0]
    h_ext = state.Constant(H_EXT)
    minimizer = MinimizerBB([MagnumDemag(), MagnumExchange(), ExternalField(lambda state: h_ext)])
    minimizer.minimize(state, maxiter=3, dm_tol=0)  # demag kernel and warm-up
    torch.cuda.synchronize()
    start = time.time()
    minimizer.minimize(state, maxiter=N_ITERATIONS, dm_tol=0)
    torch.cuda.synchronize()
    return (time.time() - start) / N_ITERATIONS


def time_relax(n, dtype):
    torch.set_default_dtype(dtype)  # the demag kernel is built in float64 and cast to this
    demag_field = DemagField(n=list(n), dx=DX, Ms=Ms, device="cuda")
    exchange_field = ExchangeField(dx=DX, A=A, Ms=Ms)
    m = start_state(n, dtype)
    h_ext = torch.tensor([H_EXT], dtype=dtype, device="cuda")
    train.DM_TOL, train.MAX_RELAX_STEPS = 0.0, 3
    train.relax(rollout.NoCorrection(), demag_field, exchange_field, m, h_ext)  # warm-up
    train.MAX_RELAX_STEPS = N_ITERATIONS
    torch.cuda.synchronize()
    start = time.time()
    train.relax(rollout.NoCorrection(), demag_field, exchange_field, m, h_ext)
    torch.cuda.synchronize()
    return (time.time() - start) / N_ITERATIONS


def main():
    torch.set_default_device("cuda")  # build the demag kernels on the GPU
    gpu = torch.cuda.get_device_name()
    print(gpu, flush=True)
    results = []
    for n in SIZES:
        for name, timer in [
            ("magnum.np MinimizerBB float64", lambda: time_magnum(n)),
            ("train.relax float64", lambda: time_relax(n, torch.float64)),
            ("train.relax float32", lambda: time_relax(n, torch.float32)),
        ]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                seconds = timer()
                entry = {"cells": n[0] * n[1] * n[2], "n": n, "method": name, "s_per_iteration": seconds,
                         "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            except torch.OutOfMemoryError:
                entry = {"cells": n[0] * n[1] * n[2], "n": n, "method": name, "s_per_iteration": "oom"}
            results.append(entry)
            print(json.dumps(entry), flush=True)
    out = Path("results/overnight")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"timing_{gpu.replace(' ', '_')}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
