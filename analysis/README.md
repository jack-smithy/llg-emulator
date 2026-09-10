# analysis/

One-off measurement scripts and their recorded outputs. These produced most of the
tables in `BENCHMARKS.md` and two of the figures in `main.tex`.

They lived in a `/tmp` session scratchpad until 2026-09-10 and were moved here
because `/tmp` does not survive a reboot, let alone a machine move. Nothing else
changed except the output paths: each script now writes its `.json` next to itself
(`Path(__file__).with_name(...)`) instead of into the current working directory.

Run everything from the repo root, on a GPU node (see `SERVER.md`). All of them
import `llg_emulator` and `datagen`, so `uv run` is required:

```bash
uv run python analysis/<script>.py [args]
```

## Scripts

| script | what it measures | writes | consumed by |
|---|---|---|---|
| `scaling.py` | s/step vs mesh size, learned integrator vs dopri5, JIT-warm, same GPU | `scaling.json` | `figures/make_scaling.py` → `scaling.pdf`; BENCHMARKS "Scaling with mesh size" |
| `solver_time.py` | wall time for neuralmag to produce each benchmark's 1 ns trajectory from stored frame 0 | stdout | BENCHMARKS "Cost: learned integrator vs reference solver" |
| `solver_repro.py` | reproducibility of the *reference* solver: two dopri5 re-runs (one exact, one 1e-6 perturbed) scored against the stored trajectory | `solver_repro.json` | BENCHMARKS "Solver self-consistency" |
| `diagnose.py <run>` | SP4 error decomposition: teacher-forced vs free-running, best time-shift, error vs \|grad m\| | `diagnose_<run>.json` | `figures/make_figures.py` Fig 4 (`error_growth.pdf`) |
| `val_by_stride.py <run>...` | per-stride one-step MSE on val films + trivial baseline | `val_by_stride.json` | BENCHMARKS "Single-step accuracy vs step size" |
| `amplify_val.py <run>...` | amplification (rollout error / one-step error) on in-distribution val trajectories | `amplify_val.json` | BENCHMARKS run-pf section |
| `profile_step.py [variant]` | where a step's time goes at 4001²: compile, demag FFT, CNN, tangent update, harness | stdout | BENCHMARKS "Where a step spends its time" |
| `large_pointwise.py <variant> <run>...` | streaming pointwise rollout error on a benchmark too big to hold in memory | stdout | BENCHMARKS "Pointwise rollout error" |

Scripts take run directories as positional arguments, e.g.:

```bash
uv run python analysis/diagnose.py weights/run1
uv run python analysis/diagnose.py weights/run-pf
uv run python analysis/val_by_stride.py weights/run-dt weights/run1
uv run python analysis/amplify_val.py weights/run1 weights/run-pf
uv run python analysis/large_pointwise.py sp4_xlarge weights/run1 weights/run-pf
```

`scaling.py` and `solver_time.py` have SLURM wrappers: `sbatch scripts/scaling.slrm`,
`sbatch scripts/solver_time.slrm`. Both need `gpu:full`.

## Recorded outputs

`scaling.json`, `solver_repro.json`, `diagnose_run1.json`, `diagnose_run-pf.json`,
`val_by_stride.json`, `amplify_val.json` are the results the current BENCHMARKS.md
tables and paper figures were built from. Keep them: `figures/make_figures.py` Fig 4
reads the two `diagnose_*` files directly rather than recomputing.

Two files are kept only as history and are **not** wired into anything:

- `runs.json` — per-epoch (epoch, train loss, val loss, sp4_bulk_rmse) for the 40-epoch
  run1 sweep; the source for `weights/run1/learning_curves.png`.
- `diagnose.json` — an earlier `diagnose.py` output, 2026-08-27. It does **not** match
  `diagnose_run1.json` and the run it came from was not recorded. Superseded; do not
  quote it.

## Caveats

- `solver_repro.json` carries a `large` key, from when `data/val/large` still existed.
  That variant has since been deleted — see `DATA.md`. Re-running `solver_repro.py`
  today fails on the missing variant unless `large` is regenerated or dropped from
  `config.BENCHMARK_VARIANTS`.
- `probe2.py`, `probe3.py`, `probe_model.py` are ad-hoc device-memory probes kept as
  the provenance for the mesh-ceiling numbers in BENCHMARKS.md. Their headline
  conclusion (5001² OOMs) was **later corrected** — it was an artefact of closing over
  `cond` as a jit constant. Read them as a record of how the wrong number arose, not
  as a current measurement.
- `scaling.py` only visits meshes whose demag tensor is already in `.nm_cache`; on a
  cold cache each new mesh costs an O(minutes) tensor build before it times anything.
