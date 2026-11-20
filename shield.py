"""The 100 x 50 x 20 um NiFe shield of the TMR study at 50 mT along y, three ways on one mesh.

uv run shield.py run --cell 2 --solver maxwell plain sigma --init vortex uniform random
uv run shield.py report

maxwell: the soft-magnet limit (chi -> infinity, |m| <= 1), unique;
plain:   micromagnetics with the real A on the coarse mesh (the study's section 8);
sigma:   the study's coarse-grained domain solver, micromagnetics with A_eff = sigma dx / 2.
Each run: <m>, the stray field at the probe points (exact, no air box) and a line cut, in
RESULTS/shield/cell<c>/<solver>_<init>.npz plus a row in summary.jsonl.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from data import RESULTS
from llg.bulk import Body, minimise_ball, minimise_sphere, normalise, stray_field
from llg.constants import MU_0

SIZE_UM = (100.0, 50.0, 20.0)
MS = 1 / MU_0  # Js = 1 T
A = 13e-12
SIGMA_BLOCH = 4.6e-3  # the study's value; 4 sqrt(A K_eff) with K_eff = mu0 Ms^2 / 2 is 9.1e-3
B_EXT = 0.05  # T, along y
PROBES_UM = [(0, y, z) for z in (12, -12, 20, -20) for y in (22, -22)]
CUT_Z_UM = np.concatenate([np.linspace(-30, -10.5, 40), np.linspace(10.5, 30, 40)])
OUT = RESULTS / "shield"


def initial_state(body: Body, kind: str, seed: int = 0) -> jnp.ndarray:
    r = body.centres()
    if kind == "vortex":  # the study's flux-closure start, m = (-y, x, 0) / r
        m = jnp.stack([-r[..., 1], r[..., 0], jnp.zeros_like(r[..., 0])], axis=-1)
        return normalise(m + 1e-9 * jnp.array([0.0, 0.0, 1.0]))  # the axis gets m = z
    if kind == "uniform":
        return jnp.zeros(body.n + (3,)).at[..., 1].set(1.0)
    if kind == "random":
        return normalise(jax.random.normal(jax.random.PRNGKey(seed), body.n + (3,)))
    raise ValueError(kind)


def solve(body: Body, solver: str, init: str, sigma: float, seed: int):
    h_ext = jnp.array([0.0, B_EXT / MU_0 / MS, 0.0])
    if solver == "maxwell":
        field = jax.jit(lambda m: body.demag(m) + h_ext)
        return minimise_ball(field, jnp.zeros(body.n + (3,)))
    a = A if solver == "plain" else sigma * body.dx[0] / 2
    field = jax.jit(lambda m: body.demag(m) + body.exchange(m, a) + h_ext)
    return minimise_sphere(field, initial_state(body, init, seed))


def run(cell_um: float, solvers: list[str], inits: list[str], sigma: float, seed: int):
    n = tuple(round(s / cell_um) for s in SIZE_UM)
    body = Body(n, (cell_um * 1e-6,) * 3, MS)  # type: ignore
    out = OUT / f"cell{cell_um:g}"
    out.mkdir(parents=True, exist_ok=True)
    probes = jnp.array(PROBES_UM) * 1e-6
    cut = jnp.stack([np.zeros_like(CUT_Z_UM), np.full_like(CUT_Z_UM, 22.0), CUT_Z_UM], -1) * 1e-6
    for solver in solvers:
        for init in inits if solver != "maxwell" else ["none"]:
            m, iters, residual = solve(body, solver, init, sigma, seed)
            to_mt = MU_0 * 1e3
            h_ext_mt = np.array([0.0, B_EXT * 1e3, 0.0])
            h_probe = np.asarray(stray_field(body, m, probes)) * to_mt + h_ext_mt
            h_cut = np.asarray(stray_field(body, m, cut)) * to_mt + h_ext_mt
            tag = solver if solver == "maxwell" else f"{solver}_{init}"
            if solver == "sigma":
                tag += f"_s{sigma * 1e3:g}"
            np.savez(out / f"{tag}.npz", m=np.asarray(m, np.float32), h_probe=h_probe, h_cut=h_cut,
                     cut_z_um=CUT_Z_UM, probes_um=np.array(PROBES_UM))  # fmt: skip
            row = {
                "cell_um": cell_um, "n": n, "solver": solver, "init": init,
                "sigma": sigma if solver == "sigma" else None, "seed": seed,
                "iters": int(iters), "residual": float(residual),
                "m_mean": np.asarray(m.mean(axis=(0, 1, 2))).tolist(),
                "hy_mt": h_probe[:, 1].tolist(), "hz_mt": h_probe[:, 2].tolist(),
            }  # fmt: skip
            with open(OUT / "summary.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{cell_um:g} um {tag}: <my> {row['m_mean'][1]:.4f}, "
                  f"Hz {np.round(h_probe[:, 2], 1)}, {int(iters)} iters, res {float(residual):.1e}",
                  flush=True)  # fmt: skip


def report():
    """Table 3 of the study, per run: <my>, |Hz| at the probes, the bridge signal
    (Hz(+22) - Hz(-22)) / 2 per sensor plane, and the probes' scatter."""
    rows = [json.loads(line) for line in (OUT / "summary.jsonl").read_text().splitlines()]
    labels = [f"({x},{y:+d},{z:+d})" for x, y, z in PROBES_UM]
    print(f"{'cell':>4} {'run':>22} {'<my>':>6} " + " ".join(f"{s:>12}" for s in labels)
          + "  bridge z=+12 -12 +20 -20   scatter12 scatter20")  # fmt: skip
    for r in rows:
        hz = np.array(r["hz_mt"])
        bridge = (hz[0::2] - hz[1::2]) / 2  # per plane z = 12, -12, 20, -20
        scatter = [np.ptp(np.abs(hz[:4])) / np.abs(hz[:4]).mean(), np.ptp(np.abs(hz[4:])) / np.abs(hz[4:]).mean()]
        name = r["solver"] + ("" if r["solver"] == "maxwell" else f"_{r['init']}")
        if r["sigma"]:
            name += f"_s{r['sigma'] * 1e3:g}"
        print(f"{r['cell_um']:>4g} {name:>22} {r['m_mean'][1]:6.4f} "
              + " ".join(f"{h:12.1f}" for h in hz)
              + "  " + " ".join(f"{b:6.1f}" for b in bridge)
              + f"   {100 * scatter[0]:8.1f}% {100 * scatter[1]:8.1f}%")  # fmt: skip


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "report"))
    parser.add_argument("--cell", type=float, default=2.0, help="cell size in um")
    parser.add_argument("--solver", nargs="*", default=["maxwell", "plain", "sigma"])
    parser.add_argument("--init", nargs="*", default=["vortex", "uniform", "random"])
    parser.add_argument("--sigma", type=float, default=SIGMA_BLOCH, help="wall energy, J/m^2")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stage == "run":
        run(args.cell, args.solver, args.init, args.sigma, args.seed)
    else:
        report()


if __name__ == "__main__":
    main()
