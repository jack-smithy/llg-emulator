# llg-emulator: learned coarse micromagnetics for NiFe shields

Soft NiFe shields (flux concentrators, e.g. 100 x 50 x 20 µm) turn an in-plane field into
out-of-plane stray-field lobes read by perpendicular TMR sensors (`Shield_Optimization_DS.pdf`).
Full micromagnetics at that size needs ~10^13 cells. This repo learns a coarse model that runs on
50-200 nm cells and is

- more accurate than coarse micromagnetics at the same cell size;
- more micromagnetic than a Maxwell M(H) solver: domains, metastability, hysteresis.

The model steps through **quasi-static field sweeps**: at each applied field it minimises the coarse
energy from the previous state,

    E(m; H) = E_demag(m) + E_exchange(m) + E_Zeeman(m; H) + E_θ(m),

with exact demag, exchange and Zeeman on the coarse grid and a learned energy density E_θ (a small
CNN) for what the coarse grid gets wrong, mainly the domain walls inside a cell.

## What works

- **Training on equilibria, not rollouts:** every stored truth state must be an equilibrium of the
  coarse model (torque loss) and a minimum, not a saddle (stability loss on the softest curvature).
  No relaxation during training.
- **Long training:** loop errors halve from 3,000 to 10,000 steps. Much longer (30,000) can fall into
  spurious minima; choosing the checkpoint by its valid rollout guarded against that (not in `train.py`
  at present: it evaluates the final model).
- **Cell size:** 50 and 100 nm reach a loop error of ~0.008 on G1 (coarse physics: 0.02-0.05);
  200 nm is ~3x worse and fails on magnets thicker than its training plates.
- **Big plates:** the 100 nm model stays on Maxwell's M(H) wherever the shield acts as a linear soft
  magnet, with 2-50 M cells on one GPU.

What did not work (switching, softness and fold losses, symmetric or context-aware energies,
implicit-gradient training through relaxations, a plain next-state CNN) is in git history before
the cleanup commit.

## Setup

```sh
uv sync    # Python 3.12: magnum.np 2.2.0, torch, h5py
```

All compute runs as Slurm jobs (see `/etc/claude-code/CLAUDE.md` for the server rules).

## Layout

| File | What |
|---|---|
| `datagen.py` | truth data: 10 nm quasi-static loops of one geometry (`G1`-`G6`, `C`), stored as 50 nm averages in `datasets/<geometry>/{train,valid,test}/loop_NN.h5` (`CLUSTER_DATAGEN.md`) |
| `physics.py` | coarse demag, exchange, effective field, batched Barzilai-Borwein relaxation, softest curvature |
| `model.py` | the learned energy `WallEnergy`, its density a `ResNet3d` |
| `nn/` | the network library: `layers.py` (convolutions with boundary modes, residual blocks with a padding mask), `arch.py` (`ResNet3d`) |
| `data.py` | coarse grid, coarsening, `StoredStates` (every stored state, padded with air to the largest grid so batches mix geometries), loading loops |
| `train.py` | training on `data.GEOMETRIES`: torque + stability, then valid and test rollouts |
| `rollout.py` | whole-loop rollouts and loop errors of a run or of coarse physics |
| `maxwell.py` | Maxwell M(H) solver (float64) |
| `experiments/loops/loops.py` | loop plots and videos on the truth: truth, coarse, learned, Maxwell |
| `experiments/big_domain/` | rollouts on big plates (up to ~100 M cells) against Maxwell, plots, videos |
| `scripts/` | Slurm job scripts: datagen (`shield.slrm` on an A100, `datagen_node.slrm` on a cluster node), training (`train.slrm`, one long job) |

## Workflow

```sh
# train at 100 nm on data.GEOMETRIES (G1, G2, G3), one long job
sbatch -J g123_s0 scripts/train.slrm g123_s0 --coarsen 2 --steps 10000 --seed 0

# test rollouts with states, coarse physics for comparison, Maxwell, plots and videos
python rollout.py g123_s0 --split test --states
python rollout.py coarse --coarsen 2 --split test --states
python experiments/loops/loops.py maxwell G1 G2 G3 --split test
python experiments/loops/loops.py plot g123_s0
python experiments/loops/loops.py movies g123_s0

# big plates, as thick as the training geometries
python experiments/big_domain/big_domain.py rollout --model g123_s0 --n 1000 500 4 --axis y --start zero --n-fields 21
python experiments/big_domain/big_domain.py rollout --model coarse --dx 100 --n 1000 500 4 --axis y --start zero --n-fields 21
python experiments/big_domain/big_domain.py maxwell --dx 100 --n 1000 500 4 --axis y
python experiments/big_domain/plot_big_domain.py loops g123_s0
```

## Data format

One folder per geometry, `<geometry>/{train,valid,test}/loop_NN.h5`, one loop per plain HDF5 file
(`datagen.save_loop`). Cell-centred, open boundaries. Per file:

- `m` (T, nx, ny, nz, 3): the 10 nm truth averaged over 5^3 cells (not renormalised);
- `B` (T, 3): applied field mu0 H per step [T];
- attributes `dx` (stored cell size), `fine_dx` (simulated cell size), `Ms`, `A`, `axis` (the loop's
  field axis, a unit vector).

Training pads every state with air (m = 0) to the largest grid among the geometries, so a batch can
mix them. The physics and the learned energy use the mask of the magnet: exchange couples only magnet
cells, padded cells never move, and the CNN zeroes them after every layer. A padded magnet therefore
behaves exactly as it does alone on its own grid, and rollouts run unpadded.

Loops from saturation have 201 steps (+200 -> -200 -> +200 mT), loops from a random start 151
(0 -> +200 -> -200 mT).

| | Magnet (µm) | Variant of the paper's shield | Cells at 10 nm | Cells at 200 nm | Time per loop (H100, estimate) |
|---|---|---|---|---|---|
| G1 | 2 × 1 × 0.4 | same proportions, small | 0.8M | 10 × 5 × 2 | 15 min |
| G2 | 4 × 2 × 0.8 | same proportions (5 : 2.5 : 1) | 6.4M | 20 × 10 × 4 | 2 h |
| G3 | 4 × 1 × 0.8 | narrower | 3.2M | 20 × 5 × 4 | 1 h |
| G4 | 4 × 4 × 0.8 | wider | 12.8M | 20 × 20 × 4 | 4 h |
| G5 | 4 × 2 × 1.6 | thicker | 12.8M | 20 × 10 × 8 | 4 h |
| G6 | 6 × 3 × 1.2 | same proportions, largest | 21.6M | 30 × 15 × 6 | 6 h |
