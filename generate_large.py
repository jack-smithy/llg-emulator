"""One permalloy LLG trajectory on the largest film a full GPU can hold, written as the
`test` split of the varied-field dataset.

Runs in the project env: `uv run generate_large.py --out datasets/permalloy_varied_field`.

Same 5x5x3 nm cells, material, dt and 1 ns as `generate_varied_field.py`, but a single
N x N film instead of 256x256. The initial condition is SP4's s-state (+x with the ends
canted, relaxed at zero field) and the applied field is SP4 field 1, both taken from
`generate_fixed_field.py`. The only thing this sample tests is size: whether an emulator
trained on 256x256 films holds up on one N/256 times wider.

N is the largest square on which *both* the magnum.np solve and a no-grad forward pass of
train.py's FNO (in-context 1, hidden 64, 2 layers, batch of one) fit on one RTX PRO 6000
(95 GiB), and the FNO is the tighter of the two. At 6144^2 the forward peaks at 73 GiB
allocated / 92 GiB reserved; 6400^2 and up run out of memory with the default allocator
(6656^2 still fits under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, at 85 GiB).
The LLG solve needs 23 GiB at 6144^2, still fits at 10240^2 (63 GiB) and only fails at
12288^2, so it is not the constraint; it takes about 1.7 s per RKF45 step at 6144^2.
`--n` overrides N.

Frames go straight into the HDF5 file as they are produced: a trajectory this size is
tens of GB, so it never exists whole in memory. How long the run took is stored in the
file's root attributes (`relax_seconds`, `relax_converged`, `llg_seconds`, `llg_steps`,
`gpu`) and printed at the end. The file is written under a `.part` suffix, which the
well's glob ignores, and renamed once complete, so a concurrent `train.py` never opens a
half-written sample.
"""

import argparse
import time
from pathlib import Path

import torch

from generate_fixed_field import DT, DX, H_EXT, MATERIAL, init_s_state
from generate_varied_field import FIELDS, H_NAMES, N_T, create_split

N = (6144, 6144, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=N[0], help="cells per side")
    args = p.parse_args()

    from magnumnp import (
        DemagField,
        ExchangeField,
        ExternalField,
        LLGSolver,
        Mesh,
        State,
        normalize,
        set_log_level,
    )

    set_log_level(250)

    n = (args.n, args.n, 1)
    state = State(Mesh(n, DX))
    state.material = dict(MATERIAL)
    state.m = normalize(init_s_state(state))
    demag, exchange = DemagField(), ExchangeField()

    t0 = time.perf_counter()
    converged = LLGSolver([demag, exchange]).relax(state)  # relax() forces alpha=1
    normalize(state.m)  # relax() steps without renormalising, unlike step()
    torch.cuda.synchronize()
    relax_seconds = time.perf_counter() - t0
    print(
        f"relaxed {n[0]}x{n[1]} in {relax_seconds:.0f} s, converged={converged}",
        flush=True,
    )

    llg = LLGSolver([demag, exchange, ExternalField(list(H_EXT))])
    state.t = 0.0
    step0 = state._step  # accepted RKF45 steps so far
    llg_seconds = 0.0

    path = args.out / "data" / "test" / "llg_test.hdf5"
    part = path.with_name(path.name + ".part")
    with create_split(part, 1, N_T, "micromagnetics_llg_varied_field", n=n) as f:
        for name, value in zip(H_NAMES, H_EXT):
            f["scalars"][name][0] = value
        for i in range(N_T):
            if i:
                t0 = time.perf_counter()
                llg.step(state, DT)
                torch.cuda.synchronize()
                llg_seconds += time.perf_counter() - t0
            frame = state.m.squeeze(2).cpu().numpy()  # (nx, ny, 3)
            for k, name in enumerate(FIELDS):
                f["t0_fields"][name][0, i] = frame[..., k]
            print(f"frame {i + 1}/{N_T}  llg {llg_seconds:.0f} s", flush=True)
        f.attrs["relax_seconds"] = relax_seconds
        f.attrs["relax_converged"] = converged
        f.attrs["llg_seconds"] = llg_seconds
        f.attrs["llg_steps"] = state._step - step0
        f.attrs["gpu"] = torch.cuda.get_device_name()
    part.replace(path)
    print(
        f"wrote {path}: relax {relax_seconds:.0f} s, 1 ns LLG solve {llg_seconds:.0f} s "
        f"in {state._step - step0} RKF45 steps on {torch.cuda.get_device_name()}"
    )


if __name__ == "__main__":
    main()
