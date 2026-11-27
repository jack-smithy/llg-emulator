"""Whole hysteresis loops on the coarse grid: each field step relaxed from the model's own previous state,
compared with the stored truth. Errors can accumulate and switching can happen at the wrong field.

    python rollout.py RUN [--split test] [--states]                     a trained run (results/RUN)
    python rollout.py coarse --coarsen 2 [--split test] [--states]      coarse physics alone

Both default to data.GEOMETRIES; --geometries G1 G3 picks others (e.g. a geometry held out of training).
Writes results/RUN/rollout_<split>_<geometries>.json (loop error per loop: mean |<m>_model - <m>_truth|
along the field axis) and, with --states, results/RUN/rollout_<split>_<geometries>.npz with every
coarse state; coarse physics writes to results/coarse_<cell size>nm/.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data import GEOMETRIES, coarse_grid, load_loops
from datagen import A, Ms
from model import WallEnergy
from physics import DEVICE, KD, DemagField, ExchangeField, NoCorrection, relax


def coarse_terms(geometry: str, factor: int) -> tuple[DemagField, ExchangeField]:
    """Demag and exchange on a geometry's coarse grid."""
    n, dx = coarse_grid(geometry, factor)
    return DemagField(n=n, dx=dx, Ms=Ms, device=DEVICE), ExchangeField(
        dx=dx, A=A, Ms=Ms
    )


def rollout(
    model, demag_field, exchange_field, m0: torch.Tensor, fields: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relax step by step from m0 (X, Y, Z, 3) through fields (T, 3): states (T, X, Y, Z, 3), converged (T,)."""
    states, converged = [m0], [True]
    m = m0[None]
    for h in fields[1:]:
        m, ok = relax(model, demag_field, exchange_field, m, h[None])
        states.append(m[0])
        converged.append(bool(ok[0]))
    return torch.stack(states), torch.tensor(converged)


def mean_along(states: torch.Tensor, axis: np.ndarray) -> np.ndarray:
    """<m> along the field axis at every step: states (T, X, Y, Z, 3) -> (T,)."""
    return states.float().mean(dim=(1, 2, 3)).cpu().numpy() @ axis


def evaluate(
    model, geometries: list[str], factor: int, split: str, keep_states: bool = False
) -> tuple[dict, dict]:
    """Roll out every loop of the split on every geometry: a report (loop errors, convergence, the
    curves) and, with keep_states, the states as float16 arrays keyed by loop."""
    report: dict = {"loops": {}}
    states = {}
    for geometry in geometries:
        demag_field, exchange_field = coarse_terms(geometry, factor)
        errors = []
        for loop in load_loops(geometry, split, factor):
            predicted, converged = rollout(
                model, demag_field, exchange_field, loop["m"][0], loop["fields"]
            )
            truth_curve = mean_along(loop["m"], loop["axis"])
            model_curve = mean_along(predicted, loop["axis"])
            errors.append(float(np.abs(model_curve - truth_curve).mean()))
            report["loops"][loop["name"]] = {
                "loop_error": errors[-1],
                "converged": float(converged.float().mean()),
                "field_along_axis": (
                    loop["fields"].cpu().numpy() @ loop["axis"]
                ).tolist(),  # [A/m]
                "truth": truth_curve.tolist(),
                "model": model_curve.tolist(),
            }
            if keep_states:
                states[loop["name"]] = predicted.cpu().numpy().astype(np.float16)
        report[f"loop_error_{geometry}"] = float(np.mean(errors))
    # every geometry weighs equally, however many loops it has
    report["loop_error"] = float(
        np.mean([report[f"loop_error_{g}"] for g in geometries])
    )
    return report, states


def load_run(run_dir: Path) -> tuple[WallEnergy, dict]:
    """A trained run's model (model.pt) and its arguments (args.json)."""
    args = json.loads((run_dir / "args.json").read_text())
    _, dx = coarse_grid(GEOMETRIES[0], args["coarsen"])
    model = WallEnergy(Kd=KD, V=math.prod(dx), Ms=Ms).to(DEVICE)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=DEVICE))
    return model.eval(), args


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run", help="a run name under results/, or 'coarse' for coarse physics alone"
    )
    parser.add_argument("--geometries", nargs="+", default=GEOMETRIES)
    parser.add_argument(
        "--coarsen", type=int, help="needed for coarse; default: the run's"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--states", action="store_true", help="also save every rolled-out state"
    )
    args = parser.parse_args()

    if args.run == "coarse":
        assert args.coarsen is not None, "coarse physics needs --coarsen"
        model, factor = NoCorrection(), args.coarsen
        _, dx = coarse_grid(args.geometries[0], factor)
        run_dir = Path("results", f"coarse_{dx[0] * 1e9:.0f}nm")
    else:
        run_dir = Path("results", args.run)
        model, run_args = load_run(run_dir)
        factor = args.coarsen or run_args["coarsen"]
    run_dir.mkdir(parents=True, exist_ok=True)

    report, states = evaluate(
        model, args.geometries, factor, args.split, keep_states=args.states
    )
    tag = "_".join(args.geometries)
    (run_dir / f"rollout_{args.split}_{tag}.json").write_text(json.dumps(report))
    if args.states:
        np.savez(
            run_dir / f"rollout_{args.split}_{tag}.npz",
            **{k.replace("/", "__"): v for k, v in states.items()},
        )
    print({k: round(v, 4) for k, v in report.items() if k.startswith("loop_error")})


if __name__ == "__main__":
    main()
