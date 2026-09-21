# A micromagnetics dataset in `the_well` format

Status: 2026-09-21.

`the_well` ([PolymathicAI](https://github.com/PolymathicAI/the_well)) is a collection of
PDE-simulation datasets with a shared HDF5 layout and a shared PyTorch loader. Putting an
LLG dataset into that layout means the standard neural-PDE baselines (FNO, U-Net, transformer
variants — whatever ships against `WellDataset`) run on micromagnetics with **no changes to
their data code**. That is the only purpose of this file's dataset: a baseline harness, not a
replacement for `data/` (which the jax surrogate in `src/llg_emulator/` trains on).

Everything lives in one script: [src/datagen/generate_well.py](src/datagen/generate_well.py).

```bash
uv run --script src/datagen/generate_well.py --device 3
uv run --script src/datagen/generate_well.py --verify-only   # re-check what's on disk
```

It is a **PEP 723 script**, not part of the package — the `# /// script` header declares its
own dependencies and `uv run --script` builds a throwaway environment for it. That is
deliberate: it uses **magnum.np**, which is a torch code, and the project env is jax. Mixing
torch's and jax's CUDA wheels in one venv is a fight with nothing to win. Nothing in the script
imports `llg_emulator`, and `pyproject.toml` is untouched.

## What is in it

muMAG standard problem #4's geometry and material, with the initial condition as the only
degree of freedom.

| | |
|---|---|
| mesh | `(100, 25, 1)` cells, `dx = (5, 5, 3) nm` — a 2D grid, z is the film thickness |
| material | permalloy: `Ms = 8e5 A/m`, `A = 1.3e-11 J/m`, `alpha = 0.02` |
| field | `H_ext = (-24.6, 4.3, 0) mT / mu_0` — SP4 field 1, the same for every trajectory |
| time | 101 frames, `dt = 10 ps`, `t_tot = 1 ns` |
| splits | `train` 48, `valid` 8, `test` 1 trajectories |

Each trajectory is relaxed to equilibrium **at zero field** from its initial condition, then
integrated under `H_ext`. Frame 0 is the relaxed state. Two initial-condition families
alternate by index:

- `init_random` — a random direction per cell, which relaxes into a multi-domain / vortex state.
- `init_uniform` — one random direction everywhere, which relaxes into a near-uniform state.

**`test` is the held-out SP4 initial condition**: the exact s-state (+x with the ends canted),
so the test split *is* standard problem 4 and its published reference curves apply. No training
trajectory starts from it.

Trajectory `i` is generated from seed `--seed + i` regardless of which split it lands in, so
growing one split cannot perturb another.

## Format decisions worth knowing

**`m` is three `t0` fields, not one `t1` field.** The obvious encoding — magnetization as a
rank-1 tensor field — is rejected by the well: `check_thewell_formatting.py` asserts a `t1`
field's trailing dimension equals `n_spatial_dims`, and the reader derives component names from
`itertools.product(spatial_dims, repeat=order)`. A 3-vector on a 2D grid has no place in that
scheme. So `mx`, `my`, `mz` go in as three scalar (`t0`) fields; the loader concatenates them
back in `field_names` order and the sample comes out 3-channel, which is what a model wants
anyway.

**Boundary conditions are `WALL` on both axes.** A thin film in vacuum is not periodic. The
loader's only supported `boundary_return_type` reduces this to a `(2, 2)` tensor of enum codes.

**`stats.yaml` is computed here, not by the well's `compute_statistics.py`**, to avoid
depending on `the_well` at generation time. It carries the six keys `ZScoreNormalization`
wants (`mean`, `std`, `rms` and the same three for the frame-to-frame delta), keyed by field
name, over the train split only.

**Times in `dimensions/time` are physical** (0 … 1e-9), not indices. The loader hands a model
`output_time_grid - input_time_grid`, so a Δt-conditioned model — the thing `src/llg_emulator/`
already does with `s_enc` — gets its step size for free without a format change.

## Verification

Both checks pass on the generated files:

```bash
# the well's own strict validator (from a clone of the_well repo)
python scripts/check_thewell_formatting.py data/well/micromagnetics_llg/data/*/*.hdf5
#   /dimensions /t0_fields /t1_fields /t2_fields /scalars /boundary_conditions passed!

# the real loader
WellDataset(path="data/well/micromagnetics_llg", well_split_name="train",
            n_steps_input=2, n_steps_output=4,
            use_normalization=True, normalization_type=ZScoreNormalization)
#   input_fields (2, 100, 25, 3)   output_fields (4, 100, 25, 3)
#   constant_scalars (6,)          boundary_conditions (2, 2)
#   space_grid (100, 25, 2)        field_names {0: ['mx','my','mz'], 1: [], 2: []}
```

`--verify-only` re-reads the written files and asserts the invariants independently of
`the_well` being installed — including that `|m| = 1` everywhere, which is the check that
caught `LLGSolver.relax()` not renormalising after its last step (unlike `step()`, which does).
Without the explicit `normalize` after relax, frame 0 of every trajectory was off unit norm by
~1e-5.

Use `WellDataset(path=...)`, **not** `well_base_path` + `well_dataset_name` — the latter asserts
the dataset name is in the hardcoded `WELL_DATASETS` list, which this one is obviously not in.

## Limits, and what to do about them

- **Only the initial condition varies.** `H_ext`, geometry and material are constant across the
  whole dataset, so a model cannot learn field conditioning from it. That was the point of a
  first pass; varying `H_ext` means moving `Hx/Hy/Hz` from constant scalars
  (`sample_varying=False`, shape `()`) to `sample_varying=True`, shape `(n_traj,)`. No other
  format change — they land in `constant_scalars` either way.
- **`Ms` and `A` ride along as constant scalars** spanning 16 orders of magnitude
  (`8e5` and `1.3e-11`). Harmless while they are constant, but nondimensionalise them before
  they start varying, or a naive baseline will choke on the input scale.
- **Frames for a split are held in RAM and written once at the end.** Fine at 48 × 101 ×
  100 × 25 × 3 float32 (≈ 145 MB); it is the wrong shape for a big run, where the frames should
  stream into the HDF5 as they are produced (`src/datagen/generate.py::run` does exactly that
  with a memory-mapped `.npy`). Killing the job partway loses the split.
- **Generation is slow for what it is.** ~1 min/trajectory on an idle GPU, but 5+ min under
  contention — the mesh is far too small to keep a GPU busy, so it is all kernel-launch overhead
  and the job is at the mercy of whoever else is on the card. `--device -1` runs on CPU; on a
  quiet machine several CPU processes in parallel would likely beat one contended GPU.
- The relax step costs more than the dynamics: `init_random` regularly needs 400–500 of
  `relax()`'s 500 `maxiter` iterations, `init_uniform` around 150.

`data/` is gitignored, so the dataset itself is not in the repo — the script is the artifact.
