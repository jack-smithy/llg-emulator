"""How much do the 'truths' differ? Loops of the same test field axes (trajectories 28-31) from:

- 10 nm vs 5 nm cells (small and medium magnet), compared after block-averaging both to 50 nm;
- 2 mT vs 4 mT field steps (small magnet, 5 nm), compared at the shared fields;
- a random kick before every field step (datagen.KICK) vs none (small magnet, 5 nm).

Loop error = mean |<m.axis>_a - <m.axis>_b| over the loop; MSE = mean |m_a - m_b|^2 per 50 nm cell
(renormalised block averages, as train.coarsen); switching = the fields of the 4 largest <m.axis> jumps.

    python experiments/overnight/compare_truths.py
"""

import json
from pathlib import Path

import h5py
import numpy as np

PAIRS = [  # (name, dataset a, cell size a [nm], dataset b, cell size b [nm], step stride of a, of b)
    ("small: 10 nm vs 5 nm", "shield_small_10nm", 10, "shield", 5, 1, 1),
    ("medium: 10 nm vs 5 nm", "shield_medium_10nm", 10, "shield_medium_5nm", 5, 1, 1),
    ("small: 2 mT vs 4 mT steps", "shield_small_2mT", 5, "shield", 5, 2, 1),
    ("small: kicked vs plain", "shield_small_kick", 5, "shield", 5, 1, 1),
]
COARSE_NM = 50


def load(dataset: str, index: int, cell_nm: int, stride: int):
    """Loop <m.axis> (T,), block-averaged m at COARSE_NM (T, X, Y, Z, 3) and the field along the axis [mT]."""
    split = "test"
    with h5py.File(Path("datasets", dataset, "data", split, f"shield_{index:02d}.hdf5")) as f:
        m = f["t1_fields/m"][0, ::stride]
        b = np.stack([f["scalars/Bx"][0, ::stride], f["scalars/By"][0, ::stride]], -1)
        angle = float(f["scalars/field_angle"][0])
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    loop = m.mean(axis=(1, 2, 3)) @ axis
    k = COARSE_NM // cell_nm
    t, x, y, z, _ = m.shape
    coarse = m.reshape(t, x // k, k, y // k, k, z // k, k, 3).mean(axis=(2, 4, 6))
    coarse /= np.linalg.norm(coarse, axis=-1, keepdims=True)
    return loop, coarse, b @ axis[:2] * 1e3


def switching_fields(loop: np.ndarray, field: np.ndarray, n: int = 4) -> list[float]:
    jumps = np.argsort(-np.abs(np.diff(loop)))[:n]
    return sorted(round(float(field[j + 1]), 1) for j in jumps)


def main():
    report = {}
    for name, data_a, cell_a, data_b, cell_b, stride_a, stride_b in PAIRS:
        if not Path("datasets", data_a).exists() or not Path("datasets", data_b).exists():
            print(f"{name}: data missing, skipped")
            continue
        rows = []
        for index in range(28, 32):
            loop_a, coarse_a, field = load(data_a, index, cell_a, stride_a)
            loop_b, coarse_b, _ = load(data_b, index, cell_b, stride_b)
            rows.append({
                "trajectory": index,
                "loop_error": float(np.abs(loop_a - loop_b).mean()),
                "mse_50nm": float(((coarse_a - coarse_b) ** 2).sum(-1).mean()),
                "switching_a_mT": switching_fields(loop_a, field),
                "switching_b_mT": switching_fields(loop_b, field),
            })
        report[name] = {"loop_error_mean": float(np.mean([r["loop_error"] for r in rows])),
                        "mse_50nm_mean": float(np.mean([r["mse_50nm"] for r in rows])), "trajectories": rows}
        print(name, json.dumps(report[name]), flush=True)
    out = Path("results/overnight")
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_truths.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
