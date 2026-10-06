"""When does micromagnetics on a coarse mesh stop working? Cell-size sweep of magnum.np
on the test films, scored pointwise and in bulk against the 5 nm reference over 10 ns.

Runs in the project env, from the repo root:

    uv run python -m datagen.coarse_breakdown run --film sq256    (GPU, magnum.np)
    uv run python -m datagen.coarse_breakdown plot --film sq256   (CPU)

`scripts/coarse_breakdown.slrm` runs the GPU stage on one MIG slice.

**Setup.** The film is one of the scaling study's test films (`eval_scaling.FILMS`,
5 nm cells, 8 trajectories: test sample j starts from `generate_varied_field.INITS[j % 3]`
relaxed, i.e. multi-domain (random per-cell start), near-uniform and s-state, each under
its own in-plane field). magnum.np is run on the same film meshed with cells from 5 nm
(a rerun of the reference solver: the control, whose error is the floor the comparison
itself has) to 160 nm (`CELLS_NM`, finer steps than the scaling study's factors of 2),
from the reference trajectory's first frame area-averaged onto that mesh and renormalised
(`regrid`, a conservative regrid that equals `eval_scaling.block_mean` at integer
ratios), for `--ns` nanoseconds. The data covers the first 1 ns; the reference is
extended past it by continuing magnum.np at 5 nm from the data's last frame.

**Recorded**, per mesh and trajectory:

- every 10 ps: bulk <m> and the exchange, demag and Zeeman energies of the coarse
  state on its own mesh (`bulk_pred`, `E_pred`); the same for the 5 nm reference
  (`bulk_truth`, `E_truth_fine`).
- at the compared steps (`steps_cmp`: every 10 ps over the first 1 ns, every 100 ps
  after), against the reference area-averaged onto the same mesh (`metrics`, named in
  the json): MSE, the well's VRMSE, the angular error's median / 90th / 99th percentile
  and the fraction of cells more than `ANGLE_DEG` off, the MSE split into textured
  cells (the area average lost norm, i.e. sub-cell structure: `TEXTURE_NORM`) and smooth
  ones, plus the persistence MSE m(t) = m(0), the regridded reference's bulk <m> and
  energies on that mesh, and over the first 1 ns the one-step error (`onestep_mse`): one
  coarse step from the regridded reference against the regridded reference, the local
  error before anything compounds.
- isotropic power spectra of the regridded reference and of the error at 1 ns and at
  the end (`spec_*`, bins of integer cycles per film side; summed, the error's gives
  3 x MSE), and float16 snapshots at 1 ns, 3 ns and the end (`snap_*`).

Written to `results-v2/<dataset>/scaling/breakdown/<film>.npz` + `.json` (rewritten
after every mesh, so a killed job keeps what it finished; `n_done` meshes are valid).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from datagen.eval_scaling import FILMS, cache_dir, trajectories
from datagen.generate_varied_field import DT, DX, MATERIAL

L_EX_NM = 5.69  # sqrt(2A / (mu0 Ms^2)) of the dataset's permalloy
# target cell sizes; the mesh is round(side / cell) cells, so they are hit exactly on
# the 1.28 um film (256 fine cells) and nearly so on the others
CELLS_NM = (5, 20 / 3, 8, 10, 80 / 7, 40 / 3, 16, 20, 160 / 7, 80 / 3, 32, 40, 160 / 3, 80, 160)
IC_TYPES = ("multi-domain", "uniform", "s-state")  # INITS[j % 3], relaxed
METRICS = ("mse", "vrmse", "ang50", "ang90", "ang99", "frac_off", "mse_textured",
           "mse_smooth", "textured_frac")  # fmt: skip
ENERGIES = ("exchange", "demag", "zeeman")
TEXTURE_NORM = 0.95  # a cell whose area-averaged m lost more norm than this is textured
ANGLE_DEG = 30.0
DATA_STEPS = 100  # frames the data has after the first


def meshes(n_fine, cells_nm=CELLS_NM):
    """Coarse mesh sides for a film of `n_fine` 5 nm cells, finest first."""
    side = n_fine * DX[0]
    out = []
    for c in cells_nm:
        n = int(round(side / (c * 1e-9)))
        if 4 <= n <= n_fine and n not in out:
            out.append(n)
    return out


def regrid_weights(n_in, n_out):
    """(n_out, n_in) weights of the conservative 1D regrid of an interval: row i is the
    fraction of coarse cell i covered by each fine cell. Rows sum to 1; at an integer
    ratio it is the block mean."""
    e_in = np.arange(n_in + 1) / n_in
    e_out = np.arange(n_out + 1) / n_out
    lo = np.maximum(e_out[:-1, None], e_in[None, :-1])
    hi = np.minimum(e_out[1:, None], e_in[None, 1:])
    return np.clip(hi - lo, 0, None) * n_out


def regrid(m, W):
    """Area-average an (N, N, 3) torch field onto the (n, N) weights' mesh, renormalise.
    Returns (m_coarse, norm): the norm of the average before renormalising, whose
    deficit is the sub-cell texture the coarse mesh cannot hold."""
    import torch

    mc = torch.einsum("ai,ijc->ajc", W, m)
    mc = torch.einsum("bj,ajc->abc", W, mc)
    norm = torch.linalg.norm(mc, dim=-1)
    return mc / norm[..., None], norm


def compared_steps(n_steps):
    return sorted(set(range(1, min(DATA_STEPS, n_steps) + 1)) | set(range(110, n_steps + 1, 10)))


def spectrum(f):
    """Isotropic power spectrum of an (n, n, 3) torch field, binned by integer |k| in
    cycles per film side, normalised so that the bins sum to the field's mean square
    per cell (|m|^2 = 1 for a unit field, 3 x MSE for an error)."""
    import torch

    n = f.shape[0]
    P = (torch.fft.fft2(f, dim=(0, 1)).abs() ** 2).sum(-1) / n**4
    k = torch.fft.fftfreq(n, d=1.0 / n, device=f.device)
    kr = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2).round().long()
    keep = kr <= n // 2
    return torch.bincount(kr[keep], weights=P[keep], minlength=n // 2 + 1).cpu().numpy()


def compare(pred, truth, norm):
    """`METRICS` of a coarse state against the regridded reference, (n, n, 3) torch."""
    import torch

    err = (pred - truth) ** 2
    var = truth.var(dim=(0, 1), unbiased=True)
    vrmse = torch.sqrt(err.mean(dim=(0, 1)) / (var + 1e-7)).mean()
    ang = torch.rad2deg(torch.arccos((pred * truth).sum(-1).clamp(-1, 1)))
    q = torch.quantile(ang.flatten(), torch.tensor([0.5, 0.9, 0.99], dtype=ang.dtype, device=ang.device))
    tex = norm < TEXTURE_NORM
    cell = err.sum(-1) / 3  # per-cell MSE in the (cells x components) convention
    nan = torch.tensor(float("nan"), dtype=cell.dtype, device=cell.device)
    return torch.stack([
        err.mean(), vrmse, q[0], q[1], q[2], (ang > ANGLE_DEG).double().mean(),
        cell[tex].mean() if tex.any() else nan,
        cell[~tex].mean() if (~tex).any() else nan,
        tex.double().mean(),
    ]).cpu().numpy()  # fmt: skip


def stage_run(out_dir, film, ns, n_traj_max, cells_nm):
    import torch
    import torch._dynamo
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
    torch._dynamo.config.cache_size_limit = 64  # one compiled exchange per mesh
    dev, dtype = torch.get_default_device(), torch.get_default_dtype()
    n_steps = int(round(ns * 1e-9 / DT))
    cmp = compared_steps(n_steps)
    cmp_index = {t: s for s, t in enumerate(cmp)}
    snap_steps = sorted({t for t in (DATA_STEPS, 300, n_steps) if t in cmp_index})
    spec_steps = sorted({t for t in (DATA_STEPS, n_steps) if t in cmp_index})

    def make_state(n, d):
        state = State(Mesh((n, n, 1), (d, d, DX[2])))
        state.material = dict(MATERIAL)
        return state

    def set_m(state, m, t=0.0):
        if not torch.is_tensor(m):
            m = torch.as_tensor(np.asarray(m), device=dev)
        state.m = m.to(dtype)[:, :, None, :].contiguous()
        state.t = float(t)

    def field(state):  # (n, n, 3) view of the state's m
        return state.m[:, :, 0, :]

    def energies(terms, state):
        return np.array([float(term.E(state)) for term in terms])

    # --- the reference: the data's 1 ns, then magnum.np at 5 nm continued from it
    paths, _ = FILMS[film]
    trajs = []
    for j, (read, h) in enumerate(trajectories(paths)):
        if j >= n_traj_max:
            break
        trajs.append((np.stack([read(t) for t in range(DATA_STEPS + 1)]), h))
    n_traj = len(trajs)
    N = trajs[0][0].shape[1]
    side = N * DX[0]
    hs = np.array([h for _, h in trajs], dtype=np.float64)
    truth = np.empty((n_traj, len(cmp), N, N, 3), np.float32)  # at the compared steps
    m0 = np.stack([frames[0] for frames, _ in trajs])
    bulk_truth = np.full((n_traj, n_steps + 1, 3), np.nan)
    E_truth_fine = np.full((n_traj, n_steps + 1, 3), np.nan)
    print(f"{film}: {n_traj} trajectories, {N}^2 cells of {DX[0] * 1e9:g} nm, "
          f"{n_steps} steps, {len(cmp)} compared", flush=True)  # fmt: skip
    fine = make_state(N, DX[0])
    demag_f, exch_f = DemagField(), ExchangeField()
    t_start = time.perf_counter()
    for j, (frames, h) in enumerate(trajs):
        terms = [exch_f, demag_f, ExternalField([float(x) for x in h])]
        for t in range(min(DATA_STEPS, n_steps) + 1):
            set_m(fine, frames[t], t * DT)
            bulk_truth[j, t] = frames[t].mean(axis=(0, 1))
            E_truth_fine[j, t] = energies(terms, fine)
            if t >= 1:
                truth[j, cmp_index[t]] = frames[t]
        if n_steps > DATA_STEPS:  # continue the reference solver past the data
            set_m(fine, frames[DATA_STEPS], DATA_STEPS * DT)
            llg = LLGSolver(terms)
            for t in range(DATA_STEPS + 1, n_steps + 1):
                llg.step(fine, DT)
                m = field(fine)
                bulk_truth[j, t] = m.mean(dim=(0, 1)).cpu().numpy()
                E_truth_fine[j, t] = energies(terms, fine)
                if t in cmp_index:
                    truth[j, cmp_index[t]] = m.float().cpu().numpy()
        print(f"  reference {j} ({IC_TYPES[j % 3]}) ready, {time.perf_counter() - t_start:.0f} s",
              flush=True)  # fmt: skip
    del trajs

    # --- the sweep
    ns_mesh = meshes(N, cells_nm)
    n_mesh = len(ns_mesh)
    shape = (n_mesh, n_traj, len(cmp))
    out = {
        "ns": np.array(ns_mesh),
        "cell_nm": side / np.array(ns_mesh) * 1e9,
        "H": hs,
        "steps_cmp": np.array(cmp),
        "snap_steps": np.array(snap_steps),
        "spec_steps": np.array(spec_steps),
        "metrics": np.full(shape + (len(METRICS),), np.nan),
        "persist_mse": np.full(shape, np.nan),
        "onestep_mse": np.full((n_mesh, n_traj, min(DATA_STEPS, n_steps)), np.nan),
        "bulk_truth_coarse": np.full(shape + (3,), np.nan),
        "E_truth_coarse": np.full(shape + (3,), np.nan),
        "bulk_pred": np.full((n_mesh, n_traj, n_steps + 1, 3), np.nan),
        "E_pred": np.full((n_mesh, n_traj, n_steps + 1, 3), np.nan),
        "seconds_per_step": np.full((n_mesh, n_traj), np.nan),
        "bulk_truth": bulk_truth,
        "E_truth_fine": E_truth_fine,
        "snap_truth_fine": np.stack(
            [truth[:, cmp_index[t]] for t in snap_steps], axis=1
        ).astype(np.float16),
    }
    meta = {
        "film": film, "fine_cells": N, "side_um": side * 1e6, "dt": DT, "n_steps": n_steps,
        "ns": ns, "n_traj": n_traj, "ic": [IC_TYPES[j % 3] for j in range(n_traj)],
        "metrics": list(METRICS), "energies": list(ENERGIES), "texture_norm": TEXTURE_NORM,
        "angle_deg": ANGLE_DEG, "l_ex_nm": L_EX_NM, "cell_nm": out["cell_nm"].tolist(),
        "n_done": 0,
    }  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)

    def save():
        np.savez(out_dir / f"{film}.npz", **out)
        (out_dir / f"{film}.json").write_text(json.dumps(meta, indent=1))

    for i, n in enumerate(ns_mesh):
        d = side / n
        W = torch.as_tensor(regrid_weights(N, n), dtype=dtype, device=dev)
        state, probe = make_state(n, d), make_state(n, d)
        demag, exch = DemagField(), ExchangeField()
        snaps_p = np.empty((n_traj, len(snap_steps), n, n, 3), np.float16)
        snaps_t = np.empty_like(snaps_p)
        spec_t = np.empty((n_traj, len(spec_steps), n // 2 + 1))
        spec_e = np.empty_like(spec_t)
        for j in range(n_traj):
            terms = [exch, demag, ExternalField([float(x) for x in hs[j]])]
            m0c, _ = regrid(torch.as_tensor(m0[j], dtype=dtype, device=dev), W)
            set_m(state, m0c)
            out["bulk_pred"][i, j, 0] = m0c.mean(dim=(0, 1)).cpu().numpy()
            out["E_pred"][i, j, 0] = energies(terms, state)
            llg, llg_probe = LLGSolver(terms), LLGSolver(terms)
            prev = m0c  # the regridded reference one step back, for the one-step error
            stepping = 0.0
            for t in range(1, n_steps + 1):
                torch.cuda.synchronize() if dev.type == "cuda" else None
                t0 = time.perf_counter()
                llg.step(state, DT)
                torch.cuda.synchronize() if dev.type == "cuda" else None
                stepping += time.perf_counter() - t0
                m = field(state)
                out["bulk_pred"][i, j, t] = m.mean(dim=(0, 1)).cpu().numpy()
                out["E_pred"][i, j, t] = energies(terms, state)
                if t not in cmp_index:
                    continue
                s = cmp_index[t]
                ref, norm = regrid(torch.as_tensor(truth[j, s], dtype=dtype, device=dev), W)
                out["metrics"][i, j, s] = compare(m, ref, norm)
                out["persist_mse"][i, j, s] = float(((m0c - ref) ** 2).mean())
                out["bulk_truth_coarse"][i, j, s] = ref.mean(dim=(0, 1)).cpu().numpy()
                set_m(probe, ref, t * DT)
                out["E_truth_coarse"][i, j, s] = energies(terms, probe)
                if t <= DATA_STEPS:
                    set_m(probe, prev, (t - 1) * DT)
                    llg_probe.step(probe, DT)
                    out["onestep_mse"][i, j, t - 1] = float(((field(probe) - ref) ** 2).mean())
                    prev = ref
                if t in snap_steps:
                    k = snap_steps.index(t)
                    snaps_p[j, k] = m.float().cpu().numpy()
                    snaps_t[j, k] = ref.float().cpu().numpy()
                if t in spec_steps:
                    k = spec_steps.index(t)
                    spec_t[j, k] = spectrum(ref)
                    spec_e[j, k] = spectrum(m - ref)
            out["seconds_per_step"][i, j] = stepping / n_steps
        out[f"snap_pred_n{n}"], out[f"snap_truth_n{n}"] = snaps_p, snaps_t
        out[f"spec_truth_n{n}"], out[f"spec_err_n{n}"] = spec_t, spec_e
        meta["n_done"] = i + 1
        save()
        mse = out["metrics"][i, :, :, 0]
        ang = out["metrics"][i, :, :, 2]
        last = len(cmp) - 1
        print(
            f"mesh {n}^2 ({d * 1e9:.1f} nm = {d * 1e9 / L_EX_NM:.2f} l_ex): mse at 1 ns "
            f"{np.nanmean(mse[:, cmp_index.get(DATA_STEPS, last)]):.2e}, at {ns:g} ns "
            f"{np.nanmean(mse[:, last]):.2e} | median angle at 1 ns "
            f"{np.nanmean(ang[:, cmp_index.get(DATA_STEPS, last)]):.1f} deg | "
            f"{np.nanmedian(out['seconds_per_step'][i]) * 1e3:.1f} ms/step | "
            f"{time.perf_counter() - t_start:.0f} s elapsed",
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stage", choices=("run", "plot"))
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--film", default="sq256", choices=list(FILMS))
    p.add_argument("--ns", type=float, default=10.0, help="trajectory length in ns")
    p.add_argument("--trajs", type=int, default=8, help="at most this many trajectories")
    p.add_argument(
        "--cells", type=float, nargs="*", default=list(CELLS_NM), help="cell sizes in nm"
    )
    args = p.parse_args()
    out_dir = cache_dir(args.dataset) / "breakdown"
    if args.stage == "run":
        stage_run(out_dir, args.film, args.ns, args.trajs, tuple(args.cells))
    else:
        from datagen.coarse_breakdown_plot import plot

        plot(out_dir, args.film)


if __name__ == "__main__":
    main()
