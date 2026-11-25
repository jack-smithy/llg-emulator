# Cluster datagen and the first training plan

The target is the paper's shield: a 100 × 50 × 20 µm NiFe block simulated with 200 nm cells. It
sees fields up to 200 mT along y, and is also tested along z. The paper starts it from a saturated
state and from demagnetised states (random or Landau).

**The plan in one sentence:** simulate smaller blocks at 10 nm on the cluster, train one correction
model at 200 nm on their block averages, then run that model at 200 nm on magnets far too big for
10 nm, up to the shield itself.

## What to generate

| | What | Why |
|---|---|---|
| Fine mesh | 10 nm everywhere | Loops at 10 nm match 5 nm. |
| Stored data | 50 nm block averages, not renormalised | 200 nm (and 100 nm) are exact averages of these. About 80 GB in total instead of 10 TB at 10 nm. |
| Geometry | Rectangular blocks with the shield's proportions, 2–6 µm long, plus narrower, wider and thicker versions (table below); one dataset folder per geometry | The shield is a thick block, not a thin film: its domains and walls are 3D. |
| Field paths | Each loop starts from saturation (±200 mT) or from a random state (0 → +200 → −200 mT), with equal chance | Random starts give the multi-domain states a loop from saturation never reaches. The paper uses both. |
| Field directions | Each loop: an in-plane axis at any angle with chance ¾, otherwise any 3D direction | The shield is driven along y and also tested along z. |
| Amount | 80 loops per geometry: 48 train, 16 validation, 16 test | Beyond about 6 loops of one magnet, more did not help; variety did. |
| Material | NiFe as now: μ0Ms = 1 T, A = 13 pJ/m, no anisotropy | The paper's material. |

### Geometries

All dimensions are multiples of 200 nm, so every coarse grid fits exactly.

| | Magnet (µm) | Variant of the paper's shield | Cells at 10 nm | Cells at 200 nm | Time per loop (H100, estimate) |
|---|---|---|---|---|---|
| G1 | 2 × 1 × 0.4 | same proportions, small | 0.8M | 10 × 5 × 2 | 15 min |
| G2 | 4 × 2 × 0.8 | same proportions (5 : 2.5 : 1) | 6.4M | 20 × 10 × 4 | 2 h |
| G3 | 4 × 1 × 0.8 | narrower | 3.2M | 20 × 5 × 4 | 1 h |
| G4 | 4 × 4 × 0.8 | wider | 12.8M | 20 × 20 × 4 | 4 h |
| G5 | 4 × 2 × 1.6 | thicker | 12.8M | 20 × 10 × 8 | 4 h |
| G6 | 6 × 3 × 1.2 | same proportions, largest | 21.6M | 30 × 15 × 6 | 6 h |

Times are for a loop from saturation (201 field steps); loops from a random state have 151 steps,
so they take about ¾ as long. The times are lower bounds until the first job measures them.

### Field paths

Every loop uses one axis and 4 mT steps.

- **From saturation:** start along the axis, then +200 → −200 → +200 mT (201 steps).
- **From a random state:** start from a random magnetisation at 0 mT, then 0 → +200 → −200 mT (151 steps).

## Running it

`datagen.py` runs one geometry and split; `python datagen.py --help` lists the settings. The
defaults are the cluster's (10 nm cells, 50 nm storage, `MinimizerBB` up to 10,000 iterations,
kick 0.01). Every loop draws its axis and start from a generator seeded by geometry, split and loop
index, so a loop is the same however the indices are shared out over workers.

1. **On the cluster:** clone the repository, create the venv as `.venv`, run `mkdir -p logs`, and
   adapt the `#SBATCH` lines of `scripts/datagen_node.slrm` (a template).
2. **Submit one node job per geometry and split** (4 workers, one per GPU):
   `sbatch scripts/datagen_node.slrm G2 train 0 12` runs loops 0–47 into `datasets/cluster/G2`.
   Validation and test: `G2 valid 0 4`, `G2 test 0 4`.
3. **Submit `G2 valid 0 4` first** and check its time per loop in `logs/worker-*.out` before submitting the rest.
4. **Split G6's train split over two jobs:** `G6 train 0 6` and `G6 train 24 6`.

Estimated time per node job (loops from saturation and from a random state mixed half and half):

| Geometry | train (12 loops per worker) | valid or test (4 loops per worker) |
|---|---|---|
| G1 | 3 h | 1 h |
| G2 | 21 h | 7 h |
| G3 | 11 h | 4 h |
| G4, G5 | 42 h | 14 h |
| G6 | two jobs of 32 h (6 loops per worker) | 21 h |

That is 19 node jobs and about 300 node-hours. A job must finish within its time limit: datagen
does not resume. Rerunning a job redoes it.

On gto4, `sbatch -J datagen scripts/shield.slrm G1 --split test --n-samples 4` runs `datagen.py`
on an A100 with any arguments (set `-t` to the loops).

## First training plan: 200 nm only, then bigger magnets

- **Train one model at 200 nm:** point `train.DATASET` at a geometry folder, e.g. `datasets/cluster/G2`, and set `train.COARSEN = 4`. Training averages the stored 50 nm data over 4 × 4 × 4 blocks and renormalises (`train.coarsen`). The cell size and grid come from the files.
- **Use it on magnets of any size at 200 nm.** The learned correction is local, and demag is computed exactly for any grid, so nothing depends on how many cells the magnet has. The shield at 200 nm is 501 × 251 × 101 = 12.7M cells, the size of the paper's own coarse simulation, so it fits on one H100.

**The main model trains on the train loops of all six geometries** and is tested on their test loops.
That error is optimistic: in one block, the shape fixes the closure domains and vortex positions,
so its test loops pass through states much like its train loops. And the shield is larger and
thicker than every block, with no reference simulation at its size. So build trust in steps, with
extra models used only as a diagnostic:

1. **Size:** train without G6 and test on all 80 G6 loops. If the loop error does not grow with magnet size, larger magnets are credible. Overnight, from the 0.5 µm to the 1 µm magnet, it did not.
2. **Thickness:** the same without G5, the thickest block (the shield is far thicker still). At 200 nm training is cheap, so leaving out each geometry in turn is affordable and shows which direction extrapolates worst.
   These models are selected on the validation loops of the geometries they train on, never on the held-out one.
3. **On the shield:**
   - ⟨m_y⟩ at 50 mT should match the Maxwell solver's 0.228;
   - Hz at the probe points should agree with the paper's coarse micromagnetics roughly to within its 10 % domain scatter;
   - the learned field should sit at walls, vortices and surfaces, and be near zero in smooth interior regions.

**Risks:**
- **200 nm is untested.** The coarsest scale tried so far is 50 nm (a factor 10 coarser than the simulation); 200 nm is a factor 20. The first question is whether the 200 nm model beats coarse physics on held-out loops. If it does not, the 50 nm data still allows training at 50 or 100 nm.
- **Surface versus interior.** The training blocks are mostly surface (at most 8 cells thick at 200 nm), while the shield is mostly interior. A magnet-mask input lets the model tell surface cells from interior ones.

## Code status

- **`datagen.py`: done.** One geometry per run, settings as command-line arguments. The files store Bx, By, Bz [T], `field_angle`, `field_elevation` and `fine_dx` (the simulated cell size).
- **Training:** `train.py`, `train_torque.py`, `train_field.py`, `rollout.py` and `benchmark.py` read the cell size and grid from the dataset. `rollout.py` and `benchmark.py` roll out each loop on its own (a folder mixes 201- and 151-step loops) and measure ⟨m⟩ along 3D axes.
- **Still missing before training on the cluster data:**
  - train on several geometry folders at once (the main model on six, the hold-out models on five);
  - add the magnet-mask input.
- `stats.yaml` is not written by the node jobs: nothing reads it (only the_well's normalisation would). If needed, once all train loops of a geometry exist:
  `.venv/bin/python -c "import well_io; well_io.write_stats('datasets/cluster/G2')"`.

## One check before production

10 nm cells were only validated on magnets 0.1 µm thick. Thick blocks contain vortex cores and
Bloch lines about as narrow as the exchange length (5 nm). Compare 10 nm with 5 nm on the small
block `C` (1 × 0.5 × 0.4 µm), with the same loops at both cell sizes (the seed does not depend on
the cell size):

    sbatch -J check10 -t 12:00:00 scripts/shield.slrm C --split test --n-samples 4 --out-dir datasets/check_10nm
    sbatch -J check5 -t 12:00:00 scripts/shield.slrm C --split test --n-samples 4 --dx 5e-9 --store-block 10 --out-dir datasets/check_5nm

At 5 nm that is 1.6M cells, about an hour per loop on a gto4 A100.

## Not covered

- **Thickness:** the real shield is 20 µm thick (100 cells at 200 nm); the thickest block here is 1.6 µm. Its bulk interior is out of reach of a 10 nm simulation.
- **The paper's disk shield** (2 µm vortex disk): non-rectangular shapes need the mask input first.
