"""How well can *any* model do? The predictability ceiling of the sq256 test films.

Runs in the project env, from the repo root, on one MIG slice (magnum.np, torch):
`uv run python -m datagen.predictability run`, then (CPU)
`uv run python -m datagen.predictability plot --runs <configuration>/seed_<s> ...`.

The reference solver itself (magnum.np at 5 nm, as `generate_varied_field`) is rerun
from each test trajectory's first frame, perturbed, and the run is coarse-grained
(`eval_scaling.block_mean`) and scored against the data exactly as the scaling study
scores a model: per-step MSE against the coarse-grained 5 nm data, mean over the 8
trajectories. Perturbations:

- `control`: none. Checks that the rerun reproduces the data, so any divergence below
  is the perturbation's.
- `eps<e>`: every cell's m gets Gaussian noise of std `e` per component (renormalised):
  the sensitivity floor. Even a near-perfect model differs from the truth by more than
  float rounding, and the switching dynamics amplify that.
- `subcell_k<k>`: noise of std `SUBCELL_STD` with its mean over every k x k block
  removed, so the coarse-grained initial state is (to first order) unchanged and only
  the sub-cell detail differs: the closure floor, i.e. how well a model that sees only
  the coarse state could do at best.

Written to `results-v2/<dataset>/scaling/predictability/`: `mse.npz` (per perturbation
and factor, the (100,) MSE curve as `<name>_k<k>` and VRMSE as `vrmse_<name>_k<k>`) and,
from `plot`, `predictability_sq256.png`: MSE (top row) and VRMSE (bottom) against step,
with each run's and the coarse solver's curves from the scaling study overlaid.
"""

import argparse
import json

import numpy as np

from datagen.eval_scaling import FILMS, N_STEPS, block_mean, cache_dir, trajectories
from datagen.eval_scaling import vrmse
from datagen.eval_scaling import run_dir
from datagen.generate_varied_field import DT, DX, MATERIAL

FILM = "sq256"
FACTORS = (2, 4, 8)
EPS = (1e-6, 1e-3)
SUBCELL_STD = 0.1  # per component, before renormalising
SEED = 0


def perturbations(rng):
    """name -> function of the fine first frame (nx, ny, 3) -> perturbed frame."""

    def normalise(m):
        return m / np.linalg.norm(m, axis=-1, keepdims=True)

    out = {"control": lambda m: m}
    for e in EPS:
        out[f"eps{e:g}"] = lambda m, e=e: normalise(
            m + e * rng.standard_normal(m.shape)
        )
    for k in FACTORS:

        def subcell(m, k=k):
            eta = SUBCELL_STD * rng.standard_normal(m.shape)
            nx, ny, _ = m.shape
            mean = eta.reshape(nx // k, k, ny // k, k, 3).mean(axis=(1, 3))
            eta -= np.repeat(np.repeat(mean, k, axis=0), k, axis=1)
            return normalise(m + eta)

        out[f"subcell_k{k}"] = subcell
    return out


def stage_run(out_dir):
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
    rng = np.random.default_rng(SEED)
    # the data, float32, coarse-grained: frames 1..100 per trajectory and factor
    refs, starts = {k: [] for k in FACTORS}, []
    for read, h in trajectories(FILMS[FILM][0]):
        frames = [read(t) for t in range(N_STEPS + 1)]
        starts.append((frames[0], h))
        for k in FACTORS:
            refs[k].append(np.stack([block_mean(f, k) for f in frames[1:]]))
    refs = {k: np.stack(v) for k, v in refs.items()}

    mse = {}
    for name, perturb in perturbations(rng).items():
        runs = {k: [] for k in FACTORS}
        for m0, h in starts:
            nx, ny, _ = m0.shape
            state = State(Mesh((nx, ny, 1), DX))
            state.material = dict(MATERIAL)
            state.m = torch.as_tensor(
                perturb(m0)[:, :, None, :], dtype=torch.get_default_dtype()
            ).to(torch.get_default_device())
            llg = LLGSolver([DemagField(), ExchangeField(), ExternalField(list(h))])
            per_k = {k: [] for k in FACTORS}
            for _ in range(N_STEPS):
                llg.step(state, DT)
                m = state.m.squeeze(2).cpu().numpy().astype(np.float32)
                for k in FACTORS:
                    per_k[k].append(block_mean(m, k))
            for k in FACTORS:
                runs[k].append(np.stack(per_k[k]))
        for k in FACTORS:
            pred = np.stack(runs[k])
            curve = ((pred - refs[k]) ** 2).mean(axis=(2, 3, 4)).mean(0)
            mse[f"{name}_k{k}"] = curve
            mse[f"vrmse_{name}_k{k}"] = vrmse(pred, refs[k])
            print(f"{name} k={k}: mse step 1 {curve[0]:.1e}, 8 {curve[7]:.1e}, "
                  f"16 {curve[15]:.1e}, 100 {curve[-1]:.1e}, mean {curve.mean():.1e}",
                  flush=True)  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "mse.npz", **mse)
    (out_dir / "config.json").write_text(
        json.dumps({"film": FILM, "eps": EPS, "subcell_std": SUBCELL_STD, "seed": SEED})
    )


def stage_plot(cache, out_dir, runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floors = dict(np.load(out_dir / "mse.npz"))
    step = np.arange(1, N_STEPS + 1)
    fig, axs = plt.subplots(
        2, len(FACTORS), figsize=(4.2 * len(FACTORS), 6.4), sharex=True, sharey="row"
    )
    colors = ("#2a78d6", "#eb6834", "#4a3aa7", "#e87ba4", "#8a8984", "#1a6b53")
    for row, (metric, prefix) in enumerate((("mse", ""), ("vrmse", "vrmse_"))):
        for ax, k in zip(axs[row], FACTORS):
            ax.plot(step, floors[f"{prefix}subcell_k{k}_k{k}"], color="#0b0b0b", lw=2.2,
                    label=f"closure floor: sub-cell noise {SUBCELL_STD:g}")  # fmt: skip
            for e, ls in zip(EPS, (":", "--")):
                ax.plot(step, floors[f"{prefix}eps{e:g}_k{k}"], color="#0b0b0b", lw=1.2,
                        ls=ls, label=f"sensitivity floor: noise {e:g}")  # fmt: skip
            ax.plot(step, floors[f"{prefix}control_k{k}"], color="#c4c3be", lw=1.0,
                    label="control (rerun, no noise)")  # fmt: skip
            base = cache / f"baseline_{FILM}_k{k}.npz"
            if base.exists() and metric in np.load(base).files:
                ax.plot(step, np.load(base)[metric], color="#1baf7a", lw=1.8,
                        label="coarse solver (magnum.np at that cell)")  # fmt: skip
            for i, run in enumerate(runs):
                f = run_dir(cache, run) / f"model_{FILM}_k{k}.npz"
                if f.exists() and metric in np.load(f).files:
                    ax.plot(step, np.load(f)[metric], color=colors[i % len(colors)],
                            lw=1.4, label=run.replace("/seed_", " seed "))  # fmt: skip
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(f"{FILM} (1.28 um), {5 * k} nm cells", fontsize=9)
            else:
                ax.set_xlabel("rollout step (10 ps)", fontsize=8)
            ax.grid(True, color="#e4e3df", lw=0.6)
        axs[row, 0].set_ylabel(
            f"{metric.upper()} vs magnum.np data (coarse-grained)", fontsize=8
        )
    handles, labels = axs[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_dir / f"predictability_{FILM}.png", dpi=150)
    print(f"wrote {out_dir / f'predictability_{FILM}.png'}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stage", choices=("run", "plot"))
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--runs", nargs="*", default=[], help="<configuration>/seed_<s>")
    args = p.parse_args()
    cache = cache_dir(args.dataset)
    out_dir = cache / "predictability"
    if args.stage == "run":
        stage_run(out_dir)
    else:
        stage_plot(cache, out_dir, args.runs)


if __name__ == "__main__":
    main()
