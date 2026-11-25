"""Datagen variants for the overnight pre-cluster tests (2026-11-24), run through datagen.main().

    python experiments/overnight/datagen_variant.py VARIANT FIRST LAST

generates the test trajectories FIRST..LAST (same field axes as datasets/shield: the axis is seeded
by the trajectory index) into datasets/shield_<VARIANT>.
"""

import sys
from pathlib import Path

import datagen

VARIANTS = {
    "medium_5nm": {"N": [200, 100, 20], "DX": [5e-9] * 3, "steps": 201},  # 1 x 0.5 x 0.1 um, 4x the cells
    "medium_10nm": {"N": [100, 50, 10], "DX": [10e-9] * 3, "steps": 201},  # the same magnet at 10 nm
    "small_10nm": {"N": [50, 25, 10], "DX": [10e-9] * 3, "steps": 201},  # datasets/shield's magnet at 10 nm
    "small_2mT": {"N": [100, 50, 20], "DX": [5e-9] * 3, "steps": 401},  # datasets/shield with 2 mT steps
    "small_kick": {"KICK": 0.01},  # datasets/shield with a random 0.01 rad kick before every field step
}

variant, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for name, value in VARIANTS[variant].items():
    setattr(datagen, name, value)
datagen.OUT_DIR = Path("datasets") / f"shield_{variant}"
datagen.SPLITS = {"test": range(first, last + 1)}
datagen.main()
