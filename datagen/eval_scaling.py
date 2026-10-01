"""Rollout accuracy and cost of the emulator across cell sizes and film sizes, against
micromagnetics run directly at the coarse cell size — the thing the emulator is for.

Runs in the project env, from the repo root, in four stages that share a cache under
`results/<dataset>/<group>/scaling/`:

    uv run python -m datagen.eval_scaling truth
    uv run python -m datagen.eval_scaling baseline
    uv run python -m datagen.eval_scaling speed
    uv run python -m datagen.eval_scaling model --runs <arch>/<configuration>/seed_<s> ...
    uv run python -m datagen.eval_scaling report --runs ...

`scripts/eval_scaling.slrm` runs the GPU stages on one MIG slice.

- **truth** (CPU): every test film (the dataset's 256^2 `test` split, the `geometries`
  films and the 6144^2 `large` film, all 5 nm cells) is cell-averaged onto each
  evaluation mesh `EVAL_FACTORS[film]` and renormalised — exactly the coarse-graining
  training uses (`utils.downsample`, a block mean at these integer factors) — and cached
  as float16 `truth_<film>_k<factor>.npy`, `(n_traj, 101, nx, ny, 3)`, with the first
  frame, every rollout's initial state, also in float32 (`first_...npy`).
- **baseline** (GPU, torch): magnum.np on the coarse mesh itself, same material and
  field, from the coarse-grained first frame for 1 ns — conventional micromagnetics at a
  cell size above the exchange length, which is what the emulator must beat. Skipped at
  factor 1, where it is the reference solver. Also the persistence baseline m(t) = m(0).
- **speed** (GPU, torch): the reference solver's own cost — magnum.np at 5 nm, which
  generated the data — per 10 ps step on each film, on the same MIG slice type.
- **solver** (GPU, JAX): neuralmag's coarse LLG step alone (`physics.CoarseLLG`, the
  base a hybrid model corrects), rolled out the same way.
- **model** (GPU, JAX): the emulator's free rollout from the same first frame, on the
  same mesh, with the demag module built for it; timed for one trajectory alone.
- **demo** (GPU, JAX): a 1 mm film at 320 nm cells, which no fine simulation can
  reach — coarse micromagnetics against each run, with no truth to compare to.

Every stage writes per-step MSE against the coarse-grained truth (mean over
trajectories), the bulk <m>(t), wall-clock seconds per 10 ps step and a few snapshot
frames to `<stage>_<film>_k<factor>.npz` (per run for `model`). `report` turns them into
`scaling/report/*.png` and `scaling/report/summary.json`.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from datagen.generate_varied_field import DT, DX, MATERIAL

FIELDS = ("mx", "my", "mz")
N_STEPS = 100  # the whole 1 ns trajectory after the first frame
SNAPSHOTS = (0, 33, 66, 99)  # rollout steps kept for figures
DATA = Path("datasets/llg_field_switching")

# test film -> (files, factors of the 5 nm cell to evaluate at). Coarse meshes stop at
# 16 cells a side; the fine end stops where one factor-1 trajectory stops fitting
# comfortably in the cache (sq512 / sq1024 at 5 nm are covered by eval_geometries).
FILMS = {
    "sq64": ([DATA / "geometries/sq64/data/test/llg_test.hdf5"], (1, 2, 4)),
    "sq128": ([DATA / "geometries/sq128/data/test/llg_test.hdf5"], (1, 2, 4, 8)),
    "sq256": (sorted((DATA / "data/test").glob("llg_test_*.hdf5")), (1, 2, 4, 8, 16)),
    "sq512": ([DATA / "geometries/sq512/data/test/llg_test.hdf5"], (2, 4, 8, 16, 32)),
    "sq1024": ([DATA / "geometries/sq1024/data/test/llg_test.hdf5"], (2, 4, 8, 16, 32)),
    "large": ([DATA / "large/llg_test_6144.hdf5"], (4, 8, 16, 32, 64)),
}


def cache_dir(dataset, group=""):
    return Path("results") / dataset / group / "scaling"


def trajectories(paths):
    """Yield (frames reader, H) per trajectory; the reader returns frame t as
    (nx, ny, 3) float32, so a 6144^2 trajectory never sits in memory whole."""
    for path in paths:
        with h5py.File(path) as f:
            for j in range(f["t0_fields"]["mx"].shape[0]):
                h = np.array([f["scalars"][c][j] for c in ("Hx", "Hy", "Hz")])
                yield (
                    lambda t, f=f, j=j: np.stack(
                        [f["t0_fields"][c][j, t] for c in FIELDS], axis=-1
                    ),
                    h,
                )


def block_mean(frame, k):
    """Cell average of a (nx, ny, 3) frame onto k-times larger cells, renormalised:
    `utils.downsample` at an integer factor."""
    nx, ny, _ = frame.shape
    m = frame.reshape(nx // k, k, ny // k, k, 3).mean(axis=(1, 3))
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def load_truth(cache, film, k):
    """The float16 reference trajectories, with the first frame in float32: the
    dynamics are sensitive enough (switching) that float16 rounding of the initial
    state alone moves a 1 ns trajectory by up to ~1e-2 MSE."""
    truth = np.load(cache / f"truth_{film}_k{k}.npy").astype(np.float32)
    truth[:, 0] = np.load(cache / f"first_{film}_k{k}.npy")
    meta = json.loads((cache / f"truth_{film}_k{k}.json").read_text())
    return truth, np.asarray(meta["H"]), tuple(meta["dx"])


def metrics(pred, truth):
    """pred, truth: (n_traj, T, nx, ny, 3) -> per-step MSE (T,) and bulk <m>(t)
    (n_traj, T, 3) of both, the mean over trajectories of the MSE."""
    mse = ((pred - truth) ** 2).mean(axis=(2, 3, 4)).mean(axis=0)
    return mse, pred.mean(axis=(2, 3)), truth.mean(axis=(2, 3))


def save_stage(path, pred, truth, seconds_per_step, keep_ref=False, **extra):
    """Metrics plus float16 snapshots of the first trajectory; the reference
    snapshots (the same for every stage) only where `keep_ref`, i.e. once, with
    the persistence baseline."""
    mse, bulk_pred, bulk_ref = metrics(pred, truth)
    snaps = {"snap_pred": pred[0, list(SNAPSHOTS)].astype(np.float16)}
    if keep_ref:
        snaps["snap_ref"] = truth[0, list(SNAPSHOTS)].astype(np.float16)
    np.savez(
        path,
        mse=mse,
        bulk_pred=bulk_pred,
        bulk_ref=bulk_ref,
        seconds_per_step=seconds_per_step,
        **snaps,
        **extra,
    )
    return mse


# --- stages -----------------------------------------------------------------


def stage_truth(cache, films):
    cache.mkdir(parents=True, exist_ok=True)
    for film in films:
        paths, factors = FILMS[film]
        # first frames in float32, the initial states of every rollout (cheap: t=0)
        missing = [
            k for k in factors if not (cache / f"first_{film}_k{k}.npy").exists()
        ]
        if missing:
            firsts = {k: [] for k in missing}
            for read, _ in trajectories(paths):
                frame = read(0)
                for k in missing:
                    firsts[k].append(block_mean(frame, k).astype(np.float32))
            for k in missing:
                np.save(cache / f"first_{film}_k{k}.npy", np.stack(firsts[k]))
        todo = [k for k in factors if not (cache / f"truth_{film}_k{k}.npy").exists()]
        if not todo:
            continue
        t0 = time.perf_counter()
        out = {k: [] for k in todo}
        hs = []
        for read, h in trajectories(paths):
            per_k = {k: [] for k in todo}
            for t in range(N_STEPS + 1):
                frame = read(t)
                for k in todo:
                    per_k[k].append(block_mean(frame, k).astype(np.float16))
            for k in todo:
                out[k].append(np.stack(per_k[k]))
            hs.append(h.tolist())
        n = frame.shape[:2]
        for k in todo:
            np.save(cache / f"truth_{film}_k{k}.npy", np.stack(out[k]))
            dx = [DX[0] * k, DX[1] * k, DX[2]]
            (cache / f"truth_{film}_k{k}.json").write_text(
                json.dumps({"H": hs, "dx": dx, "fine_cells": list(n)})
            )
        print(f"truth {film}: {todo} in {time.perf_counter() - t0:.0f} s", flush=True)


def stage_baseline(cache, films):
    import torch
    from magnumnp import (
        DemagField,
        ExchangeField,
        ExternalField,
        LLGSolver,
        Mesh,
        State,
        set_log_level,
    )

    set_log_level(250)
    for film in films:
        for k in FILMS[film][1]:
            out = cache / f"baseline_{film}_k{k}.npz"
            persist = cache / f"persistence_{film}_k{k}.npz"
            truth, hs, dx = load_truth(cache, film, k)
            ref = truth[:, 1:]
            if not persist.exists():
                persisted = np.repeat(truth[:, :1], N_STEPS, axis=1)
                save_stage(persist, persisted, ref, 0.0, keep_ref=True)
            if k == 1 or out.exists():
                continue
            preds, seconds = [], []
            for m0, h in zip(truth[:, 0], hs):
                nx, ny, _ = m0.shape
                state = State(Mesh((nx, ny, 1), dx))
                state.material = dict(MATERIAL)
                state.m = torch.as_tensor(
                    m0[:, :, None, :], dtype=torch.get_default_dtype()
                ).to(torch.get_default_device())
                llg = LLGSolver([DemagField(), ExchangeField(), ExternalField(list(h))])
                frames = []
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(N_STEPS):
                    llg.step(state, DT)
                    frames.append(state.m.squeeze(2).cpu().numpy().astype(np.float32))
                seconds.append((time.perf_counter() - t0) / N_STEPS)
                preds.append(np.stack(frames))
            mse = save_stage(out, np.stack(preds), ref, float(np.median(seconds)))
            print(
                f"baseline {film} k={k} ({nx}x{ny}): rollout mse mean {mse.mean():.2e} "
                f"final {mse[-1]:.2e}, {np.median(seconds):.3f} s/step",
                flush=True,
            )


def stage_speed(cache, films, n_time=5):
    """Wall-clock of the reference solver that generated the data: magnum.np at 5 nm
    on each film, from a real trajectory's first frame under its own field, seconds
    per 10 ps (median of `n_time` steps after one warm-up). The 6144^2 film does not
    fit a MIG slice; its generator's own timings (full GPU) are read from the file.
    Written to `speed_solver.json`."""
    import torch
    from magnumnp import (
        DemagField,
        ExchangeField,
        ExternalField,
        LLGSolver,
        Mesh,
        State,
        set_log_level,
    )

    set_log_level(250)
    out = cache / "speed_solver.json"
    rows = json.loads(out.read_text()) if out.exists() else {}
    for film in films:
        paths = FILMS[film][0]
        if film == "large":
            with h5py.File(paths[0]) as f:
                a = dict(f.attrs)
            rows[film] = {
                "cells": 6144**2,
                "seconds_per_step": float(a["llg_seconds"]) / N_STEPS,
                "relax_seconds": float(a["relax_seconds"]),
                "device": f"{a['gpu']} (full GPU, recorded at generation)",
            }
            continue
        trajs = trajectories(paths)  # held, so its file stays open while read
        read, h = next(trajs)
        m0 = read(0)
        trajs.close()
        nx, ny, _ = m0.shape
        state = State(Mesh((nx, ny, 1), DX))
        state.material = dict(MATERIAL)
        state.m = torch.as_tensor(
            m0[:, :, None, :], dtype=torch.get_default_dtype()
        ).to(torch.get_default_device())
        llg = LLGSolver([DemagField(), ExchangeField(), ExternalField(list(h))])
        llg.step(state, DT)
        seconds = []
        for _ in range(n_time):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llg.step(state, DT)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - t0)
        rows[film] = {
            "cells": nx * ny,
            "seconds_per_step": float(np.median(seconds)),
            "device": torch.cuda.get_device_name() + " (MIG slice)",
        }
        print(f"speed {film}: {np.median(seconds):.3f} s per 10 ps at 5 nm", flush=True)
        out.write_text(json.dumps(rows, indent=1))
    out.write_text(json.dumps(rows, indent=1))


def stage_solver(cache, films):
    """neuralmag's coarse LLG step alone (`physics.CoarseLLG`, the base of a hybrid
    model) rolled out on each coarse mesh from the same first frame: what the
    learned correction is added to. Written to `nmsolver_<film>_k<factor>.npz`."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from physics import llg_cache

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")

    @eqx.filter_jit
    def roll(solver, m0, h):
        def step(m, _):
            m = solver(m, h)
            return m, m

        return jax.lax.scan(step, m0, None, length=N_STEPS)[1]

    for film in films:
        for k in FILMS[film][1]:
            out = cache / f"nmsolver_{film}_k{k}.npz"
            if k == 1 or out.exists():
                continue
            truth, hs, dx = load_truth(cache, film, k)
            solver = llg_cache(truth.shape[2:4], tuple(dx))
            preds, seconds = [], []
            for m0, h in zip(truth[:, 0], hs):
                args = (
                    solver,
                    jnp.moveaxis(jnp.asarray(m0), -1, 0),
                    jnp.asarray(h) / MATERIAL["Ms"],
                )
                if not seconds:
                    np.asarray(roll(*args))  # compile for this mesh
                t0 = time.perf_counter()
                preds.append(np.moveaxis(np.asarray(roll(*args)), 1, -1))
                seconds.append((time.perf_counter() - t0) / N_STEPS)
            mse = save_stage(
                out, np.stack(preds), truth[:, 1:], float(np.median(seconds))
            )
            print(f"neuralmag solver {film} k={k}: rollout mse mean {mse.mean():.2e} "
                  f"final {mse[-1]:.2e}, {np.median(seconds) * 1e3:.2f} ms/step", flush=True)  # fmt: skip


def model_speed(model, config, sizes=(64, 128, 256, 512, 1024, 2048)):
    """Seconds per 10 ps step of one trajectory at 5 nm on an N^2 film, for the
    speed comparison with the reference solver on the same mesh: the emulator's
    cost does not depend on the state, so a random one will do."""
    import jax.numpy as jnp
    import jax.random as jr

    from physics import demag_cache, llg_cache
    from utils import conditioning, rollout

    rows = {}
    for n in sizes:
        m = jr.normal(jr.PRNGKey(0), (1, 1, n, n, 3))
        m = m / jnp.linalg.norm(m, axis=-1, keepdims=True)
        x = (jnp.arange(n) + 0.5) * DX[0]
        grid = jnp.stack(jnp.meshgrid(x, x, indexing="ij"), axis=-1)[None]
        scalars = jnp.array([[MATERIAL["Ms"], MATERIAL["A"], 0.02, 2e4, 1e4, 0.0]])
        cond = conditioning(scalars, grid if config.cell_size_cond else None)
        demag = demag_cache((n, n), DX) if config.use_demag else None
        solver = llg_cache((n, n), DX) if config.use_solver else None
        args = (model, m, cond, grid, 10, demag, config.coords, solver)
        np.asarray(rollout(*args))
        t0 = time.perf_counter()
        np.asarray(rollout(*args))
        rows[n] = {"cells": n * n, "seconds_per_step": (time.perf_counter() - t0) / 10}
        print(f"emulator speed {n}^2: {rows[n]['seconds_per_step'] * 1e3:.2f} ms/step")
    return rows


def stage_model(
    cache,
    films,
    runs,
    dataset,
    results_root="results",
    suffix="",
    group="",
    speed=True,
):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from model import load_model
    from physics import demag_cache, llg_cache
    from utils import conditioning, rollout

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    for run in runs:
        path = Path(results_root) / dataset / group / run
        model, config = load_model(path, key=jr.PRNGKey(0), tag="model")
        model = eqx.nn.inference_mode(model)
        tag = run.replace("/", "__") + suffix
        speed_run, speed = speed, cache / f"speed_model_{tag}.json"
        if speed_run and not speed.exists():
            speed.write_text(json.dumps(model_speed(model, config), indent=1))
        for film in films:
            for k in FILMS[film][1]:
                out = cache / f"model_{tag}_{film}_k{k}.npz"
                if out.exists():
                    continue
                truth, hs, dx = load_truth(cache, film, k)
                n_traj, _, nx, ny, _ = truth.shape
                x = (jnp.arange(nx) + 0.5) * dx[0]
                y = (jnp.arange(ny) + 0.5) * dx[1]
                grid1 = jnp.stack(jnp.meshgrid(x, y, indexing="ij"), axis=-1)[None]
                demag = demag_cache((nx, ny), tuple(dx)) if config.use_demag else None
                solver = llg_cache((nx, ny), tuple(dx)) if config.use_solver else None
                scalars = np.zeros((n_traj, 6), np.float32)
                scalars[:, 0] = MATERIAL["Ms"]
                scalars[:, 1] = MATERIAL["A"]
                scalars[:, 2] = MATERIAL["alpha"]
                scalars[:, 3:6] = hs
                # batch the trajectories while a batch stays below ~8 x 256^2 cells
                bs = int(max(1, min(n_traj, 8 * 256**2 // (nx * ny))))
                preds, seconds = [], []
                for i in range(0, n_traj, bs):
                    sl = slice(i, min(i + bs, n_traj))
                    b = sl.stop - sl.start
                    grid = jnp.repeat(grid1, b, axis=0)
                    cond = conditioning(
                        jnp.asarray(scalars[sl]),
                        grid if config.cell_size_cond else None,
                    )
                    x0 = jnp.asarray(truth[sl, :1])

                    args = (
                        model, x0, cond, grid, N_STEPS, demag, config.coords, solver
                    )  # fmt: skip
                    if i == 0:
                        np.asarray(rollout(*args))  # compile for this mesh
                    t0 = time.perf_counter()
                    preds.append(np.asarray(rollout(*args)))
                    seconds.append((time.perf_counter() - t0) / (N_STEPS * b))
                # one trajectory alone, as the solver runs, for the speed comparison
                args = (
                    model,
                    x0[:1],
                    cond[:1],
                    grid[:1],
                    N_STEPS,
                    demag,
                    config.coords,
                    solver,
                )
                np.asarray(rollout(*args))
                t0 = time.perf_counter()
                np.asarray(rollout(*args))
                single = (time.perf_counter() - t0) / N_STEPS
                mse = save_stage(
                    out,
                    np.concatenate(preds),
                    truth[:, 1:],
                    single,
                    batched_seconds_per_step=float(np.median(seconds)),
                )
                print(
                    f"model {run} {film} k={k} ({nx}x{ny}): rollout mse mean "
                    f"{mse.mean():.2e} final {mse[-1]:.2e}, "
                    f"{single * 1e3:.2f} ms/step single, "
                    f"{np.median(seconds) * 1e3:.2f} ms/step/traj batched",
                    flush=True,
                )


def tiled_network(model, x, cond, tile=1024, halo=48):
    """`model.ResidualEmulator` applied tile by tile, for meshes whose activations do
    not fit one device: each tile is computed with `halo` extra cells on every side
    (more than the dilated ResNet's 44-cell receptive field) and cropped, so for a
    strictly local network (`use_norm=False`) the result equals the whole-mesh one.
    With GroupNorm it would only approximate it, the statistics being per tile.

    x: (C, nx, ny) as `utils.model_inputs` assembles one sample; cond: (n_cond,).
    """
    import equinox as eqx
    import jax.numpy as jnp

    net = eqx.filter_jit(lambda model, x, c: model.network(x, meta_data=c))
    _, nx, ny = x.shape
    dm = jnp.zeros((3, nx, ny), x.dtype)
    for i0 in range(0, nx, tile):
        for j0 in range(0, ny, tile):
            a0, b0 = max(0, i0 - halo), max(0, j0 - halo)
            a1, b1 = min(nx, i0 + tile + halo), min(ny, j0 + tile + halo)
            y = net(model, x[:, a0:a1, b0:b1], cond)
            ti, tj = min(tile, nx - i0), min(tile, ny - j0)
            dm = dm.at[:, i0 : i0 + ti, j0 : j0 + tj].set(
                y[:, i0 - a0 : i0 - a0 + ti, j0 - b0 : j0 - b0 + tj]
            )
    m1 = x[:3] + model.output_scale * dm
    return m1 / jnp.linalg.norm(m1, axis=0, keepdims=True)


def stage_demo(cache, runs, dataset, group, side_um=1000.0, cell_nm=320.0):
    """A film no fine simulation can reach: `side_um` square, `cell_nm` cells, from a
    uniform +x state relaxed on that mesh (alpha = 1, zero field, 2 ns), then 1 ns
    under the 30.7 um film's own field — coarse micromagnetics alone against each
    run's rollout. There is no truth; the 30.7 um film's truth, a smaller film of the
    same material under the same field, is the physical reference for the interior.
    Steps run one at a time and the network tile by tile (`tiled_network`), so
    memory stays at one tile's activations. Written to
    `demo_<side>um_<cell>nm.npz`: bulk <m>(t) of the whole film and of its interior
    (10 um from the edges), seconds per step, and 5x-subsampled snapshots.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from model import load_model
    from physics import demag_cache, llg_cache
    from utils import conditioning, model_inputs, spatial_features

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    n = round(side_um * 1e3 / cell_nm)
    d = side_um * 1e-6 / n
    dx = (d, d, DX[2])
    h = np.asarray(json.loads((cache / "truth_large_k16.json").read_text())["H"][0])
    print(f"demo: {side_um:g} um film, {n}^2 cells of {d * 1e9:.0f} nm", flush=True)
    demag = demag_cache((n, n), dx)
    solver = llg_cache((n, n), dx)
    relaxer = llg_cache((n, n), dx, alpha=1.0)
    step_relax = eqx.filter_jit(lambda s, m: s(m, jnp.zeros(3)))
    step_solver = eqx.filter_jit(lambda s, m, h: s(m, h))
    m = jnp.zeros((3, n, n)).at[0].set(1.0)
    t0 = time.perf_counter()
    for _ in range(200):
        m = step_relax(relaxer, m)
    m.block_until_ready()
    print(f"  relaxed in {time.perf_counter() - t0:.0f} s", flush=True)
    m_relaxed = m

    edge = int(np.ceil(10e-6 / d))
    out = {"h": h, "cells": n, "cell_nm": d * 1e9, "edge_cells": edge}

    def record(name, frames_fn):
        bulk, inner, snap, seconds = [], [], [], []
        m = m_relaxed
        for t in range(N_STEPS):
            t0 = time.perf_counter()
            m = frames_fn(m)
            m.block_until_ready()
            seconds.append(time.perf_counter() - t0)
            bulk.append(np.asarray(m.mean(axis=(1, 2))))
            inner.append(np.asarray(m[:, edge:-edge, edge:-edge].mean(axis=(1, 2))))
            if t in SNAPSHOTS:
                snap.append(np.asarray(m[:, ::5, ::5]).astype(np.float16))
        out[f"{name}_bulk"] = np.stack(bulk)
        out[f"{name}_inner"] = np.stack(inner)
        out[f"{name}_snap"] = np.stack(snap)
        out[f"{name}_seconds_per_step"] = float(np.median(seconds[1:]))
        print(f"  {name}: {np.median(seconds[1:]):.3f} s/step, <m>(1 ns) "
              f"{np.round(bulk[-1], 3)}", flush=True)  # fmt: skip

    hj = jnp.asarray(h / MATERIAL["Ms"])
    record("solver", lambda m: step_solver(solver, m, hj))
    x = (jnp.arange(n) + 0.5) * d
    grid = jnp.stack(jnp.meshgrid(x, x, indexing="ij"), axis=-1)[None]
    scalars = jnp.asarray(
        [[MATERIAL["Ms"], MATERIAL["A"], MATERIAL["alpha"], *h]], dtype=jnp.float32
    )
    for run in runs:
        model, config = load_model(
            Path("results") / dataset / group / run, jr.PRNGKey(0), "model"
        )
        model = eqx.nn.inference_mode(model)
        cond = conditioning(scalars, grid if config.cell_size_cond else None)
        feats = spatial_features(grid, config.coords)
        dm = demag if config.use_demag else None
        sv = solver if config.use_solver else None

        if config.use_norm:
            print("  NB GroupNorm: tiled inference only approximates this model")

        def step_model(m, model=model, feats=feats, dm=dm, sv=sv, cond=cond):
            x = model_inputs(m[None], feats, dm, sv, cond)[0]
            return tiled_network(model, x, cond[0])

        record(run.replace("/", "__"), step_model)
    np.savez(cache / f"demo_{side_um:g}um_{cell_nm:g}nm.npz", **out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "stage",
        choices=("truth", "baseline", "speed", "solver", "model", "demo", "report"),
    )
    p.add_argument("--dataset", default="llg_field_switching")
    # runs and the cache live under results/<dataset>/<group>/
    p.add_argument("--group", default="")
    p.add_argument("--films", nargs="*", default=list(FILMS))
    p.add_argument(
        "--runs", nargs="*", default=[], help="<arch>/<configuration>/seed_<s>"
    )
    # model stage: where the runs live, and a tag suffix for evaluating a checkpoint
    # without taking the final model's cache slot
    p.add_argument("--results-root", default="results")
    p.add_argument("--suffix", default="")
    # demo stage: the film side and cell size
    p.add_argument("--side-um", type=float, default=1000.0)
    p.add_argument("--cell-nm", type=float, default=320.0)
    p.add_argument(
        "--no-speed", action="store_true", help="skip the model stage's speed sweep"
    )
    args = p.parse_args()
    cache = cache_dir(args.dataset, args.group)
    if args.stage == "truth":
        stage_truth(cache, args.films)
    elif args.stage == "baseline":
        stage_baseline(cache, args.films)
    elif args.stage == "speed":
        stage_speed(cache, args.films)
    elif args.stage == "solver":
        stage_solver(cache, args.films)
    elif args.stage == "demo":
        stage_demo(
            cache, args.runs, args.dataset, args.group, args.side_um, args.cell_nm
        )
    elif args.stage == "model":
        stage_model(
            cache,
            args.films,
            args.runs,
            args.dataset,
            args.results_root,
            args.suffix,
            args.group,
            not args.no_speed,
        )
    else:
        from datagen.scaling_report import report

        report(cache, args.films, args.runs)


if __name__ == "__main__":
    main()
