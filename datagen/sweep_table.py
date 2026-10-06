"""Rollout scores of several runs on one film, for choosing between them without looking at
the test films: `uv run python -m datagen.sweep_table --film valid256 --runs k1/seed_0 ...`

Per run and cell size, from the scaling cache's `model_<film>_k<k>.npz` (the mean over the
film's trajectories of the per-step MSE against the 5 nm reference): the MSE averaged over
the 100 steps and at step 100, with the coarse LLG solver's (`baseline_<film>_k<k>.npz`)
for reference, and a one-number score per run, the geometric mean over the cell sizes of
the run's mean MSE over the solver's (< 1 beats it; at 5 nm, where the solver is the
reference itself, the run's mean MSE is listed alone). Also written as
`<cache>/report/sweep_<film>.json`.
"""

import argparse
import json

import numpy as np

from datagen.eval_scaling import FILMS, cache_dir, run_dir
from datagen.scaling_report import CELL_NM


def table(cache, film, runs, cells=None):
    ks = [k for k in FILMS[film][1] if cells is None or CELL_NM * k in cells]
    solver = {}
    for k in ks:
        f = cache / f"baseline_{film}_k{k}.npz"
        solver[k] = np.load(f)["mse"] if f.exists() else None
    rows = {}
    for run in runs:
        rows[run] = {}
        for k in ks:
            f = run_dir(cache, run) / f"model_{film}_k{k}.npz"
            if not f.exists():
                continue
            mse = np.load(f)["mse"]
            rows[run][k] = {"mean": float(np.nanmean(mse)), "final": float(mse[-1])}
            if solver[k] is not None:
                rows[run][k]["ratio"] = float(np.nanmean(mse) / solver[k].mean())
    return ks, solver, rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="llg_field_switching")
    p.add_argument("--film", default="valid256", choices=list(FILMS))
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--cells", type=float, nargs="+", default=None, help="cell sizes (nm) to list and score")
    args = p.parse_args()
    cache = cache_dir(args.dataset)
    ks, solver, rows = table(cache, args.film, args.runs, args.cells)
    cells = [CELL_NM * k for k in ks]
    print(f"{args.film}: mean MSE over 100 steps (MSE at step 100), x1e-3; ratio = run / LLG solver")
    print(f"{'run':>24} " + " ".join(f"{c:>15g}" for c in cells) + "   score")
    line = f"{'LLG solver':>24} "
    for k in ks:
        line += f"{'reference':>15} " if solver[k] is None else f"{1e3 * solver[k].mean():7.2f} ({1e3 * solver[k][-1]:5.2f}) "
    print(line)
    out = {"cells_nm": cells, "runs": {}}
    for run, r in rows.items():
        line = f"{run:>24} "
        ratios = []
        for k in ks:
            if k not in r:
                line += f"{'-':>15} "
                continue
            line += f"{1e3 * r[k]['mean']:7.2f} ({1e3 * r[k]['final']:5.2f}) "
            if "ratio" in r[k]:
                ratios.append(r[k]["ratio"])
        score = float(np.exp(np.mean(np.log(ratios)))) if ratios else float("nan")
        print(line + f"  {score:6.3f}")
        out["runs"][run] = {"score": score, "cells": {str(k): v for k, v in r.items()}}
    (cache / "report").mkdir(parents=True, exist_ok=True)
    tag = "" if args.cells is None else "_to" + f"{max(args.cells):g}nm"
    (cache / "report" / f"sweep_{args.film}{tag}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
