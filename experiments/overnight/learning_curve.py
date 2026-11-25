"""Learning curve of the coarse correction (energy + stability term, train_torque.py): how does the test
loop error fall with the number of training trajectories? Each run trains, then rolls out the test loops
(rollout.py). The 50 nm, 24-trajectory point is results/torque_energy_stable_50nm.

    python experiments/overnight/learning_curve.py
"""

from pathlib import Path

import rollout
import train
import train_torque

RUNS = [(5, 6), (5, 12), (5, 24), (10, 6), (10, 12)]  # (COARSEN, training trajectories)

train_torque.ENERGY = True
train_torque.STABILITY_WEIGHT = 1.0
for coarsen, n_trajectories in RUNS:
    run_dir = Path("results/overnight") / f"lc_{coarsen * 5}nm_{n_trajectories}traj"
    if (run_dir / "rollout_test.json").exists():
        continue
    train.COARSEN = coarsen
    train_torque.N_TRAIN_TRAJECTORIES = n_trajectories
    train_torque.RUN_DIR = rollout.RUN_DIR = run_dir
    train_torque.main()
    rollout.main()
