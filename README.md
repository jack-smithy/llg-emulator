# llg-emulator: learned coarse micromagnetics for NiFe shields

Soft NiFe shields (flux concentrators, e.g. 100 x 50 x 20 um) turn an in-plane field into
out-of-plane stray-field lobes read by perpendicular TMR sensors (`Shield_Optimization_DS.pdf`).
Full micromagnetics at that size needs ~10^13 cells. The goal is a learned coarse model that
runs at 0.5-2 um cells and is:

- more accurate than coarse LLG at the same cell size;
- more micromagnetic than a Maxwell M(H) solver: domains, metastability, hysteresis.

The model steps through **quasi-static field sweeps**: one step is the relaxed state after one
field step, so the field history lives in the state.

**Training truth:** 5 nm micromagnetics (magnum.np) of small elements, block-averaged to
coarse cells, stored in [the Well](https://github.com/PolymathicAI/the_well)'s HDF5 format.

**Status:**
- step 1 of the plan (repo skeleton, shared modules, tests) is done;
- the Maxwell baseline (B2) is written. No data generated yet.

## Setup

```sh
uv sync    # Python 3.12: magnum.np 2.2.0, torch 2.13.0, the_well 1.2.0, h5py
```

## Layout

- `shield/geometry.py` — NiFe constants; plate shapes (prism, ellipse) as cell masks
- `shield/protocols.py` — field histories (major loops, reversal curves, operating point, AC
  demagnetisation) and initial states
- `shield/coarsen.py` — block averaging with torch's `avg_pool3d` (moment-conserving, composable)
- `shield/strayfield.py` — stray field via magnum.np's FFT demag on an air-padded grid, probe
  points, scipy interpolation
- `shield/well_io.py` — write Well files
- `baselines/maxwell.py` — the Maxwell M(H) solver:
  - the PDF's saturating law, χ-continuation;
  - Newton's method on the convex magnetostatic energy, with scipy's CG for each step, in float64.
- `baselines/check_maxwell_pdf.py` — reproduce the PDF's Maxwell numbers on the 100 × 50 × 20 µm
  shield
- `tests/` — checks, e.g. the stray field against magpylib's analytic cuboid, and the Maxwell
  solver against an exact single-cell solution and the forward law

Importing magnum.np (e.g. through `shield.strayfield`) switches torch to float64 and makes the
GPU the default device.

## Data format

One dataset folder per tier, `data/{train,valid,test}/*.hdf5`. All files of a split share one
grid. Cell-centred, centred on the origin; open boundaries. Per file:

- `t1_fields/m` (n, T, nx, ny, nz, 3): moment density in units of Ms (|m| <= 1, 0 in air)
- `t0_fields/mask` (n, nx, ny, nz): magnetic volume fraction (constant in time)
- `t0_fields/e_ex` (n, T, nx, ny, nz): exchange energy density of the fine state
- `scalars`:
  - per step: applied field Bx, By, Bz (T), energies, probe fields;
  - per trajectory: shape sizes;
  - constants: Ms, A, ...
- `dimensions/time`: the field-step index

## Tests

Run inside a job, never in the login shell:

```sh
srun -p devcpu -c 4 -t 15 uv run pytest -q
```
