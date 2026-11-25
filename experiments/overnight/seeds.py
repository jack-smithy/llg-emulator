"""Run-to-run spread of the coarse correction: repeat runs of train_torque.py with other seeds (initial
weights and sample order; the training subset is fixed), each followed by its test rollout.

    python experiments/overnight/seeds.py
"""

from pathlib import Path

import rollout
import train
import train_torque

RUNS = [  # (COARSEN, stability weight, training trajectories, seed); seed 0 of each is an earlier run
    (5, 0.0, 24, 1), (5, 0.0, 24, 2),  # seed 0: results/torque_energy_25nm
    (5, 1.0, 24, 1),  # seed 0: results/overnight/lc_25nm_24traj
    (10, 1.0, 24, 1), (10, 1.0, 24, 2),  # seed 0: results/torque_energy_stable_50nm
    (10, 1.0, 12, 1),  # seed 0: results/overnight/lc_50nm_12traj
]

train_torque.ENERGY = True
for coarsen, weight, n_trajectories, seed in RUNS:
    run_dir = Path("results/overnight") / f"seed_{coarsen * 5}nm_w{weight:g}_{n_trajectories}traj_s{seed}"
    if (run_dir / "rollout_test.json").exists():
        continue
    train.COARSEN = coarsen
    train_torque.STABILITY_WEIGHT, train_torque.N_TRAIN_TRAJECTORIES, train_torque.SEED = weight, n_trajectories, seed
    train_torque.RUN_DIR = rollout.RUN_DIR = run_dir
    train_torque.main()
    rollout.main()
