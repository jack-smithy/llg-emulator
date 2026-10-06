"""Rollout accuracy and cost of the emulator across cell sizes and film sizes, against
micromagnetics run directly at the coarse cell size — the thing the emulator is for.

Runs in the project env, from the repo root, in stages that share a cache under
`results-v2/<dataset>/scaling/` (`paths.RESULTS`: another root via `RESULTS_ROOT`):

    uv run python -m datagen.eval_scaling truth
    uv run python -m datagen.eval_scaling baseline
    uv run python -m datagen.eval_scaling speed
    uv run python -m datagen.eval_scaling model --runs <configuration>/seed_<s> ...
    uv run python -m datagen.eval_scaling report --runs ...

`scripts/eval_scaling.slrm` runs the GPU stages on one MIG slice.

- **truth** (CPU): every test film (`FILMS`: the dataset's 256^2 test shards, the
  held-out geometries and the 6144^2 `large` film, all 5 nm cells) is cell-averaged onto each
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
- **model** (GPU, JAX): the emulator's free rollout from the same first frame, on the
  same mesh; timed for one trajectory alone.
- **demo** (GPU, JAX): a 1 mm film at 320 nm cells, which no fine simulation can
  reach — coarse micromagnetics against each run, with no truth to compare to.

Every stage writes per-step MSE against the coarse-grained truth (mean over
trajectories), the bulk <m>(t), wall-clock seconds per 10 ps step and a few snapshot
frames to `<stage>_<film>_k<factor>.npz`. The model-independent stages (truth,
baseline, speed) share the cache; everything a run produces (`model`, `demo`, its
report) goes to that run's own `results-v2/<dataset>/<run>/scaling/`. `report` turns them into
`scaling/report/*.png` and `scaling/report/summary.json`.
"""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np

from datagen.generate_varied_field import DT, DX, MATERIAL
from paths import RESULTS

FIELDS = ("mx", "my", "mz")
N_STEPS = 100  # the whole 1 ns trajectory after the first frame
SNAPSHOTS = (0, 33, 66, 99)  # rollout steps kept for figures
DATA = Path("datasets/llg_field_switching")

# test film -> (files, factors of the 5 nm cell to evaluate at). Coarse meshes stop at
# 16 cells a side; the fine end stops where one factor-1 trajectory stops fitting
# comfortably in the cache (sq512 / sq1024 at 5 nm are covered by eval_geometries).
# Every film is one file of the dataset's test split, data/test/llg_test_<name>.hdf5,
# except sq256: the generator's numbered shards llg_test_<i>.hdf5.
TEST = DATA / "data/test"
FILMS = {
    "sq64": ([TEST / "llg_test_sq64.hdf5"], (1, 2, 4)),
    "sq128": ([TEST / "llg_test_sq128.hdf5"], (1, 2, 4, 8)),
    "sq256": (sorted(TEST.glob("llg_test_[0-9]*.hdf5")), (1, 2, 4, 8, 16)),
    "sq512": ([TEST / "llg_test_sq512.hdf5"], (2, 4, 8, 16, 32)),
    "sq1024": ([TEST / "llg_test_sq1024.hdf5"], (2, 4, 8, 16, 32)),
    "large": ([TEST / "llg_test_large6144.hdf5"], (4, 8, 16, 32, 64)),
    # the validation split's 256^2 films: rollout scores to select models on without
    # touching the test films
    "valid256": (
        sorted((DATA / "data/valid").glob("llg_valid_[0-9]*.hdf5")),
        (1, 2, 4, 8, 16),
    ),
}


def cache_dir(dataset):
    return RESULTS / dataset / "scaling"


def run_dir(cache, run):
    """Where one run's scaling outputs go: `results-v2/<dataset>/<run>/scaling/`."""
    return Path(cache).parent / run / "scaling"


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


def vrmse(pred, truth, eps=1e-7):
    """the well's VRMSE per step: sqrt(spatial MSE / (spatial variance of the truth
    + eps)) per trajectory, step and component (`the_well...VRMSE`, unbiased std),
    then the mean over components and trajectories. (n_traj, T, nx, ny, 3) -> (T,)."""
    err = ((pred - truth) ** 2).mean(axis=(2, 3))
    var = truth.std(axis=(2, 3), ddof=1) ** 2
    return np.sqrt(err / (var + eps)).mean(axis=(0, 2))


def metrics(pred, truth):
    """pred, truth: (n_traj, T, nx, ny, 3) -> per-step MSE and VRMSE (T,) and bulk
    <m>(t) (n_traj, T, 3) of both, MSE and VRMSE averaged over trajectories."""
    mse = ((pred - truth) ** 2).mean(axis=(2, 3, 4)).mean(axis=0)
    return mse, vrmse(pred, truth), pred.mean(axis=(2, 3)), truth.mean(axis=(2, 3))


def done(path):
    """A stage output that exists and already carries every metric (outputs from
    before `vrmse` was recorded are redone)."""
    return path.exists() and "vrmse" in np.load(path).files


def save_stage(path, pred, truth, seconds_per_step, keep_ref=False, context=0, **extra):
    """Metrics plus float16 snapshots of the first trajectory; the reference
    snapshots (the same for every stage) only where `keep_ref`, i.e. once, with
    the persistence baseline. The first `context` steps of pred are the true frames a
    multi-frame model was given, not predictions: their MSE and VRMSE are NaN."""
    mse, vrmse_, bulk_pred, bulk_ref = metrics(pred, truth)
    mse[:context] = np.nan
    vrmse_[:context] = np.nan
    snaps = {"snap_pred": pred[0, list(SNAPSHOTS)].astype(np.float16)}
    if keep_ref:
        snaps["snap_ref"] = truth[0, list(SNAPSHOTS)].astype(np.float16)
    np.savez(
        path,
        mse=mse,
        vrmse=vrmse_,
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
            if not done(persist):
                persisted = np.repeat(truth[:, :1], N_STEPS, axis=1)
                save_stage(persist, persisted, ref, 0.0, keep_ref=True)
            if k == 1 or done(out):
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


def model_speed(model, config, sizes=(64, 128, 256, 512, 1024, 2048)):
    """Seconds per 10 ps step of one trajectory at 5 nm on an N^2 film, for the
    speed comparison with the reference solver on the same mesh: the emulator's
    cost does not depend on the state, so a random one will do. A solver-in-the-loop
    model's cost includes its coarse solver, a demag model's its demag field."""
    import jax.numpy as jnp
    import jax.random as jr

    from physics import mesh_physics
    from utils import conditioning, rollout

    rows = {}
    for n in sizes:
        m = jr.normal(jr.PRNGKey(0), (1, config.in_frames, n, n, 3))
        m = m / jnp.linalg.norm(m, axis=-1, keepdims=True)
        x = (jnp.arange(n) + 0.5) * DX[0]
        grid = jnp.stack(jnp.meshgrid(x, x, indexing="ij"), axis=-1)[None]
        scalars = jnp.array([[MATERIAL["Ms"], MATERIAL["A"], 0.02, 2e4, 1e4, 0.0]])
        physics = mesh_physics(config, (n, n), DX)
        args = (
            model,
            m,
            conditioning(scalars, grid),
            grid,
            10,
            scalars[:, 3:6],
            physics,
        )
        np.asarray(rollout(*args))
        t0 = time.perf_counter()
        np.asarray(rollout(*args))
        rows[n] = {"cells": n * n, "seconds_per_step": (time.perf_counter() - t0) / 10}
        print(f"emulator speed {n}^2: {rows[n]['seconds_per_step'] * 1e3:.2f} ms/step")
    return rows


def stage_model(cache, films, runs, dataset, speed=True):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from model import load_config, load_model
    from physics import mesh_physics
    from utils import conditioning, rollout

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    for run in runs:
        path = RESULTS / dataset / run
        model = eqx.nn.inference_mode(load_model(path, jr.PRNGKey(0), "model"))
        config = load_config(path)
        # a multi-frame model starts from the first n_in true frames
        n_in = config.in_frames
        n_steps = N_STEPS - (n_in - 1)
        rdir = run_dir(cache, run)
        rdir.mkdir(parents=True, exist_ok=True)
        speed_file = rdir / "speed_model.json"
        if speed and not speed_file.exists():
            speed_file.write_text(json.dumps(model_speed(model, config), indent=1))
        for film in films:
            for k in FILMS[film][1]:
                out = rdir / f"model_{film}_k{k}.npz"
                if done(out):
                    continue
                truth, hs, dx = load_truth(cache, film, k)
                n_traj, _, nx, ny, _ = truth.shape
                x = (jnp.arange(nx) + 0.5) * dx[0]
                y = (jnp.arange(ny) + 0.5) * dx[1]
                grid1 = jnp.stack(jnp.meshgrid(x, y, indexing="ij"), axis=-1)[None]
                physics = mesh_physics(config, (nx, ny), dx)
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
                    cond = conditioning(jnp.asarray(scalars[sl]), grid)
                    x0 = jnp.asarray(truth[sl, :n_in])
                    h = jnp.asarray(hs[sl], jnp.float32)
                    args = (model, x0, cond, grid, n_steps, h, physics)
                    if i == 0:
                        np.asarray(rollout(*args))  # compile for this mesh
                    t0 = time.perf_counter()
                    # context frames first, so pred lines up with truth[:, 1:]
                    pred = np.asarray(rollout(*args))
                    preds.append(np.concatenate([truth[sl, 1:n_in], pred], axis=1))
                    seconds.append((time.perf_counter() - t0) / (n_steps * b))
                # one trajectory alone, as the solver runs, for the speed comparison
                args = (model, x0[:1], cond[:1], grid[:1], n_steps, h[:1], physics)
                np.asarray(rollout(*args))
                t0 = time.perf_counter()
                np.asarray(rollout(*args))
                single = (time.perf_counter() - t0) / n_steps
                mse = save_stage(
                    out,
                    np.concatenate(preds),
                    truth[:, 1:],
                    single,
                    context=n_in - 1,
                    batched_seconds_per_step=float(np.median(seconds)),
                )
                print(
                    f"model {run} {film} k={k} ({nx}x{ny}): rollout mse mean "
                    f"{np.nanmean(mse):.2e} final {mse[-1]:.2e}, "
                    f"{single * 1e3:.2f} ms/step single, "
                    f"{np.median(seconds) * 1e3:.2f} ms/step/traj batched",
                    flush=True,
                )


def tiled_network(model, x, cond, halo, tile=1024):
    """The model's network applied tile by tile, for meshes whose activations do not
    fit one device: each tile is computed with `halo` extra cells on every side (at
    least the network's receptive radius, `model.receptive_radius`) and cropped, so
    for the strictly local network (no GroupNorm) the result equals the whole-mesh
    one.

    x: (C, nx, ny), one sample's network input (`utils.with_coords`, the coarse step
    P(m) of a solver-in-the-loop model, or `ClosureEmulator.features`); cond: (3,).
    returns: the network's output (3, nx, ny).
    """
    import equinox as eqx
    import jax.numpy as jnp

    net = eqx.filter_jit(lambda model, x, c: model.network(x, meta_data=c))
    _, nx, ny = x.shape
    out = jnp.zeros((3, nx, ny), x.dtype)
    for i0 in range(0, nx, tile):
        for j0 in range(0, ny, tile):
            a0, b0 = max(0, i0 - halo), max(0, j0 - halo)
            a1, b1 = min(nx, i0 + tile + halo), min(ny, j0 + tile + halo)
            y = net(model, x[:, a0:a1, b0:b1], cond)
            ti, tj = min(tile, nx - i0), min(tile, ny - j0)
            out = out.at[:, i0 : i0 + ti, j0 : j0 + tj].set(
                y[:, i0 - a0 : i0 - a0 + ti, j0 - b0 : j0 - b0 + tj]
            )
    return out


def stage_demo(cache, runs, dataset, side_um=1000.0, cell_nm=320.0):
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

    from model import load_config, load_model, receptive_radius
    from physics import LLGStepper, demag_cache
    from utils import conditioning, with_coords

    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    n = round(side_um * 1e3 / cell_nm)
    d = side_um * 1e-6 / n
    dx = (d, d, DX[2])
    h = np.asarray(json.loads((cache / "truth_large_k16.json").read_text())["H"][0])
    print(f"demo: {side_um:g} um film, {n}^2 cells of {d * 1e9:.0f} nm", flush=True)
    solver = LLGStepper((n, n), dx, DT, **MATERIAL)
    relaxer = LLGStepper((n, n), dx, DT, **(MATERIAL | {"alpha": 1.0}))
    step_relax = eqx.filter_jit(lambda s, m: s(m, jnp.zeros(3)))
    step_solver = eqx.filter_jit(lambda s, m, h: s(m, h))
    # the demag module of the demo mesh, built only if a use_demag run asks for it
    demag_field = eqx.filter_jit(lambda d, m: d(m))
    demag_mesh = []
    m = jnp.zeros((3, n, n)).at[0].set(1.0)
    t0 = time.perf_counter()
    for _ in range(200):
        m = step_relax(relaxer, m)
    m.block_until_ready()
    print(f"  relaxed in {time.perf_counter() - t0:.0f} s", flush=True)
    m_relaxed = m

    edge = int(np.ceil(10e-6 / d))
    out = {"h": h, "cells": n, "cell_nm": d * 1e9, "edge_cells": edge}

    def record(out, name, frames_fn):
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

    hj = jnp.asarray(h)  # A/m
    # coarse micromagnetics once; every run's file carries it next to its rollout
    record(out, "solver", lambda m: step_solver(solver, m, hj))
    x = (jnp.arange(n) + 0.5) * d
    grid = jnp.stack(jnp.meshgrid(x, x, indexing="ij"), axis=-1)[None]
    scalars = jnp.asarray(
        [[MATERIAL["Ms"], MATERIAL["A"], MATERIAL["alpha"], *h]], dtype=jnp.float32
    )
    cond = conditioning(scalars, grid)
    for run in runs:
        path = RESULTS / dataset / run
        model = eqx.nn.inference_mode(load_model(path, jr.PRNGKey(0), "model"))
        halo = receptive_radius(model)
        config = load_config(path)
        if config.in_frames > 1:
            # the demo film starts from one relaxed state: no frame history to give
            print(f"  skipping {run}: needs {config.in_frames} frames of context")
            continue

        def step_model(m, model=model, halo=halo, config=config):
            if config.closure:  # the solver under the network's closure field
                x, meta = model.features(m, cond[0], solver)
                h_theta = (
                    model.eps(cond[0]) * solver.Ms * tiled_network(model, x, meta, halo)
                )
                return step_solver(solver, m, hj[:, None, None] + h_theta)
            if config.solver_in_the_loop:  # the network corrects the coarse step P(m)
                x = step_solver(solver, m, hj)
            else:
                x = with_coords(m[None], grid)[0]
                if config.use_demag:  # the whole film's field, then tiled with the rest
                    if not demag_mesh:
                        demag_mesh.append(demag_cache((n, n), dx))
                    x = jnp.concatenate([x, demag_field(demag_mesh[0], m)])
            m1 = x[:3] + tiled_network(model, x, cond[0], halo)
            return m1 / jnp.linalg.norm(m1, axis=0, keepdims=True)

        run_out = dict(out)
        record(run_out, run.replace("/", "__"), step_model)
        rdir = run_dir(cache, run)
        rdir.mkdir(parents=True, exist_ok=True)
        np.savez(rdir / f"demo_{side_um:g}um_{cell_nm:g}nm.npz", **run_out)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "stage",
        choices=("truth", "baseline", "speed", "model", "demo", "report"),
    )
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--films", nargs="*", default=list(FILMS))
    p.add_argument("--runs", nargs="*", default=[], help="<configuration>/seed_<s>")
    # demo stage: the film side and cell size
    p.add_argument("--side-um", type=float, default=1000.0)
    p.add_argument("--cell-nm", type=float, default=320.0)
    p.add_argument(
        "--no-speed", action="store_true", help="skip the model stage's speed sweep"
    )
    # report stage: where the figures go; default the run's own scaling/report for
    # one run, <cache>/report for a comparison of several
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cache = cache_dir(args.dataset)
    if args.stage == "truth":
        stage_truth(cache, args.films)
    elif args.stage == "baseline":
        stage_baseline(cache, args.films)
    elif args.stage == "speed":
        stage_speed(cache, args.films)
    elif args.stage == "demo":
        stage_demo(cache, args.runs, args.dataset, args.side_um, args.cell_nm)
    elif args.stage == "model":
        stage_model(cache, args.films, args.runs, args.dataset, not args.no_speed)
    else:
        from datagen.scaling_report import report

        report(cache, args.films, args.runs, args.out)


if __name__ == "__main__":
    main()
