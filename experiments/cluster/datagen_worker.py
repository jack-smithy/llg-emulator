"""One datagen worker on the big cluster: one process per GPU, sharing a dataset's trajectories by index.

    python experiments/cluster/datagen_worker.py WORKER N_WORKERS

runs the trajectories i with i % N_WORKERS == WORKER through datagen.main(). When all workers are done:
python -c "import well_io; well_io.write_stats('datasets/shield_cluster')". The geometry below is a
placeholder until the target is decided (see results/overnight/REPORT.md).
"""

import sys
from pathlib import Path

import datagen

# --- to decide: the overnight tests support 10 nm cells and 4 mT steps; see REPORT.md ---
datagen.DX = [10e-9] * 3
datagen.N = [400, 200, 40]  # 4 x 2 x 0.4 um: 20 x 10 x 2 cells of 200 nm
datagen.steps = 201  # 4 mT steps from +200 to -200 mT and back
datagen.MAXITER = 10000  # the 4x larger magnet hit 2000 in 8-13 % of its steps at 5 nm
datagen.OUT_DIR = Path("datasets/shield_cluster")
datagen.WRITE_STATS = False  # each worker sees only its share: write stats once all are done
N_TRAJECTORIES = {"train": 96, "valid": 8, "test": 8}
# -------------------------------------------------------------------------------------------

worker, n_workers = int(sys.argv[1]), int(sys.argv[2])
first, splits = 0, {}
for split, count in N_TRAJECTORIES.items():
    splits[split] = [i for i in range(first, first + count) if i % n_workers == worker]
    first += count
datagen.SPLITS = splits
datagen.main()
