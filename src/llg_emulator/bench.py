"""Benchmark harness: evaluate a trained LLGEmulator on the SP4 target + val set.

Headline metric is the standard-problem-4 deliverable: the bulk magnetization
trajectory <m>(t) from an autoregressive rollout vs the micromagnetic reference.
We also report the field-space correlation-vs-step (rollout stability) and the
one-step val MSE (the training objective on held-out data).

Usage:
    uv run python -m llg_emulator.bench <weights.eqx> [--tag NAME]

Writes/updates BENCHMARKS.md (a table, one row per tagged run) so results are
diffable and easy to read. Model architecture is read from the sidecar
<weights>.json written by save_model().
"""

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from llg_emulator.config import dataset_dir
from llg_emulator.data import LLGStepperSource, load_trajectory
from llg_emulator.metrics import correlation, correlation_epoch
from llg_emulator.model import LLGEmulator, ModelConfig
from llg_emulator.rollout import rollout_trajectory
from llg_emulator.training import loss_fn

BENCH_FILE = Path("BENCHMARKS.md")


def save_model(model: LLGEmulator, config: ModelConfig, path: Path):
    """Serialise weights + a sidecar json holding the arch needed to rebuild."""
    path = Path(path)
    eqx.tree_serialise_leaves(path, model)
    meta = {"hidden_channels": config.hidden_channels, "num_blocks": config.num_blocks}
    path.with_suffix(".json").write_text(json.dumps(meta))


def load_model(path: Path) -> LLGEmulator:
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    config = ModelConfig(
        hidden_channels=meta["hidden_channels"], num_blocks=meta["num_blocks"]
    )
    skeleton = LLGEmulator(config=config, key=jr.PRNGKey(0))
    return eqx.tree_deserialise_leaves(path, skeleton)


def sp4_metrics(model: LLGEmulator, size: str = "small") -> dict:
    """Autoregressive rollout on the SP4 reference; bulk-<m> RMSE + correlation."""
    model = eqx.nn.inference_mode(model)
    path = dataset_dir("sp4", size) / "sample_0"
    m_true, H = load_trajectory(path)
    m_true = jnp.asarray(m_true)
    m_pred = rollout_trajectory(model, m_true, jnp.asarray(H), include_init=True)

    # bulk magnetization <m>(t): (t, 3)
    bulk_true = jnp.mean(m_true, axis=(2, 3))
    bulk_pred = jnp.mean(m_pred, axis=(2, 3))
    bulk_rmse = float(jnp.sqrt(jnp.mean((bulk_true - bulk_pred) ** 2)))
    # per-component final-frame error (the switched state)
    bulk_final_err = np.abs(np.asarray(bulk_true[-1] - bulk_pred[-1]))

    # field correlation vs step: 1.0 = perfect alignment
    corr, _ = correlation(m_pred[None], m_true[None])  # add batch axis
    corr = np.asarray(corr)

    return {
        "sp4_bulk_rmse": bulk_rmse,
        "sp4_corr_mean": float(corr.mean()),
        "sp4_corr_final": float(corr[-1]),
        "sp4_bulk_final_err": bulk_final_err.tolist(),
    }


def val_metrics(model: LLGEmulator, size: str = "small") -> dict:
    """One-step MSE + rollout correlation on the val split."""
    model = eqx.nn.inference_mode(model)
    val = LLGStepperSource(dataset_dir("val", size))

    # one-step MSE over the whole val set, chunked so the conv stack fits in memory
    bs = 8
    total, n = 0.0, 0
    for s in range(0, len(val), bs):
        idx = range(s, min(s + bs, len(val)))
        m0 = jnp.stack([val[i]["m0"] for i in idx])
        m1 = jnp.stack([val[i]["m1"] for i in idx])
        H = jnp.stack([val[i]["H"] for i in idx])
        total += float(loss_fn(model, m0, m1, H)) * len(idx)
        n += len(idx)
    step_mse = total / n

    corr_mean, _ = correlation_epoch(model, val)
    corr_mean = np.asarray(corr_mean)
    return {
        "val_step_mse": step_mse,
        "val_corr_mean": float(corr_mean.mean()),
        "val_corr_final": float(corr_mean[-1]),
    }


def evaluate(model: LLGEmulator, size: str = "small") -> dict:
    return {**sp4_metrics(model, size), **val_metrics(model, size)}


def append_benchmark(tag: str, metrics: dict, extra: dict | None = None):
    """Append one row to BENCHMARKS.md (creates the table on first write)."""
    extra = extra or {}
    cols = [
        "tag",
        "sp4_bulk_rmse",
        "sp4_corr_mean",
        "sp4_corr_final",
        "val_step_mse",
        "val_corr_mean",
        "val_corr_final",
        "notes",
    ]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4e}" if abs(v) < 1e-2 else f"{v:.4f}"
        return str(v)

    row_vals = [tag]
    row_vals += [fmt(metrics[c]) for c in cols[1:-1]]
    row_vals.append(extra.get("notes", ""))
    row = "| " + " | ".join(row_vals) + " |"

    if not BENCH_FILE.exists():
        BENCH_FILE.write_text(
            "# SP4 emulator benchmarks\n\n"
            "Lower `sp4_bulk_rmse` and higher `*_corr_*` (→1.0) are better. "
            "`sp4_*` = autoregressive rollout on the standard-problem-4 target "
            "(out-of-distribution field). `val_*` = held-out one-step / rollout.\n\n"
            + header + "\n" + sep + "\n"
        )
    with BENCH_FILE.open("a") as f:
        f.write(row + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=str)
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--size", type=str, default="small")
    ap.add_argument("--notes", type=str, default="")
    args = ap.parse_args()

    model = load_model(Path(args.weights))
    metrics = evaluate(model, args.size)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    append_benchmark(args.tag, metrics, {"notes": args.notes})
    print(f"\nAppended '{args.tag}' to {BENCH_FILE}")


if __name__ == "__main__":
    main()
