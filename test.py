"""Smoke-test the training + eval pipeline end to end.

Runs `main.py` for a single gradient step and `eval.py` on the result,
exercising the same CLI surface the slurm jobs use, inside a temporary results
directory that exists only while the script runs — a test leaves no trace in
`results/`.

Run from the repo root: `uv run python test.py [--arch fno --no-use-demag]`;
`scripts/test.slrm` runs it on one MIG slice and `make test` submits it.
"""

import subprocess
import sys
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--arch", type=str, default="dilated")
parser.add_argument("--dataset", type=str, default="llg_field_switching")
parser.add_argument("--use-demag", action=BooleanOptionalAction, default=True)
parser.add_argument("--coords", default="absolute")
parser.add_argument("--cell-size-cond", action=BooleanOptionalAction, default=False)
parser.add_argument("--pool-factors", type=float, nargs="+", default=[1.0])
parser.add_argument("--unroll", type=int, default=1)
parser.add_argument("--use-solver", action=BooleanOptionalAction, default=False)
parser.add_argument("--use-norm", action=BooleanOptionalAction, default=True)
args = parser.parse_args()

COMMON = (
    "--seed", "0",
    "--arch", args.arch,
    "--configuration", "smoke",
    "--dataset", args.dataset,
)  # fmt: skip


def run(script, *extra):
    cmd = [sys.executable, script, *COMMON, *extra]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


with tempfile.TemporaryDirectory(prefix="llg-smoke-") as tmp:
    run(
        "main.py",
        "--results-root", tmp,
        "--num-steps", "1",
        "--batch-size", "4",
        "--num-workers", "1",
        "--hidden-channels", "8",
        "--num-blocks", "2",
        "--num-modes", "8",
        "--pool-factors", *map(str, args.pool_factors),
        "--unroll", str(args.unroll),
        "--use-solver" if args.use_solver else "--no-use-solver",
        "--use-norm" if args.use_norm else "--no-use-norm",
        "--use-demag" if args.use_demag else "--no-use-demag",
        "--coords", args.coords,
        "--cell-size-cond" if args.cell_size_cond else "--no-cell-size-cond",
    )  # fmt: skip
    # eval takes one factor per run; score the first of the training set
    run(
        "eval.py",
        "--results-root", tmp,
        "--max-batches", "2",
        "--pool-factor", str(args.pool_factors[0]),
    )  # fmt: skip

    run_dir = Path(tmp) / args.dataset / args.arch / "smoke" / "seed_0"
    expected = (
        "model.eqx", "metadata.json", "stats.json", "rollout.npy",
        "learning_curve.png", "rollout.png", "norms.png",
        "ref.gif", "pred.gif", "err.gif",
    )  # fmt: skip
    missing = [name for name in expected if not (run_dir / name).exists()]
    assert not missing, f"missing artifacts: {missing}"

print("smoke test passed")
