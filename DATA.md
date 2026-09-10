# Regenerating the data

Status: 2026-09-10. Written while moving the repo off `gto4`.

`data/` is 66 GB and gitignored. This file records what is in it, what can be
regenerated, and what cannot — so the dataset does not have to be treated as a
precious object that must be copied byte-for-byte forever.

**The short version.** Every sample's `metadata.json` records the `PRNGKey` it was
generated from, and *everything else about that sample derives from that key*. All 642
of them are archived in **`manifest/metadata.zip`** (247 KB, committed), so a bare clone
carries enough information to rebuild the entire training set. It costs about
**4 GPU-hours**. The frames themselves are the only thing you would copy to save time.

Restore the metadata into an empty `data/` before regenerating — paths inside the zip
are relative to the repo root, so it unpacks straight into place:

```bash
unzip manifest/metadata.zip -d .        # -> data/<split>/<variant>/sample-*/metadata.json
```

Rebuild the archive after generating new samples (run from the repo root):

```bash
find data -name metadata.json | sort | zip -q -X -9 manifest/metadata.zip -@
```

`data/` as a whole stays gitignored; the zip is the only copy in version control.

The two deleted benchmark variants are the exception — their metadata went with them.
See [Deleted variants](#deleted-variants).

---

## What is on disk

| split / variant | trajectories | mesh `n` (cells) | `m.npy` shape (nodal) | size |
|---|---|---|---|---|
| `train/fixed_geo` | 512 | (256, 256) | (101, 257, 257, 3) | 77 MB each, 39 GB |
| `val/fixed_geo`   | 128 | (256, 256) | (101, 257, 257, 3) | 77 MB each, 9.9 GB |
| `val/sp4`         | 1   | (100, 25)   | (101, 101, 26, 3)   | 1.2 MB |
| `val/sp4_xlarge`  | 1   | (4000, 4000)| (101, 4001, 4001, 3)| 19 GB |

All share: `dx = (5e-9, 5e-9, 3e-9)`, `dt = 10 ps`, `t_tot = 1 ns` (101 frames),
material `{Ms: 8e5, A: 1.3e-11, alpha: 0.02}`.

`val/fixed_geo` sample dirs are named `sample-000NN-of-00256` but only 128 exist:
the split was cut from 256 to 128 on 2026-08-28 to reclaim disk quota, and the
survivors (`00001`–`00128`) kept their original names. The 128 deleted ones are gone
along with their metadata. This does not matter — the 128 that remain are what every
recorded result was trained and validated against.

## Why `generate.py` cannot regenerate the training set as-is

Three separate problems in [src/datagen/generate.py](src/datagen/generate.py). None is
hard to fix, but none of them is obvious from reading the file, which is the point of
writing them down.

**1. The entrypoint only builds SP4.** `__main__` calls `main_sp4()` and nothing else.
`main_fixed_geo` and `main_variable_geo` are unreachable from
`python -m datagen.generate`.

**2. The two `main_*` functions are misnamed — they do the opposite of what they say.**

| function | writes to | mesh it builds | what it actually is |
|---|---|---|---|
| `main_fixed_geo` | `data/<split>/xlarge` | hardcoded `(2000, 2000)` | the generator for the **`large`** benchmark |
| `main_variable_geo` | `data/<split>/fixed_geo` | random `n ∈ [16, 512)` per sample | **nothing that was ever used** |

So `main_variable_geo` writes into the directory holding the training set, but produces
a *different mesh per sample*. `LLGStepperSource` rejects a split with mixed meshes
(it batches, and a batch of mixed grids has no shape), so running it would produce a
dataset that cannot be trained on. The 256×256 fixed-mesh training set on disk was
built by an earlier version of this file that no longer exists in git history.

**3. Off-by-one in the key indexing.** Both `main_*` functions do:

```python
keys = jr.split(key, args.n_samples)
for index in range(args.start, args.n_samples + args.start):
    ... generate_sample(keys[index], ...)
```

`index` is 1-based but `keys` has length `n_samples`, so `keys[0]` is never used and
the final iteration indexes out of range. Any fix should use `keys[index - 1]`, or
better, drop the shared array and derive each key as `jr.fold_in(key, index)` so
`--start` and `--n-samples` stop changing what earlier samples contain.

## Regenerating `train/fixed_geo` and `val/fixed_geo` exactly

Each sample is fully determined by its recorded key. Verified 2026-09-10 on six
samples spanning both splits: feeding `metadata.json["key"]` back through
`jr.split` → `random_H_ext` / `random_init` reproduces the stored `H_ext` and `init`
label exactly, in all six cases.

The chain inside `generate_sample` is:

```python
key_h, key_init = jr.split(key)          # key == metadata.json["key"]
init_label, init_m_fn = random_init(key_init)
H_ext = random_H_ext(key_h)
```

So the regeneration loop is: for every sample dir, read its `metadata.json`, and call
`generate_sample` with that key and that mesh. Something like this, as a new
`main_*` in `generate.py`:

```python
def main_from_manifest():
    """Rebuild samples from their metadata.json (unzip the manifest first).
    Idempotent: skips any sample whose m.npy already exists, so it can be
    re-run after an interruption."""
    parser = ArgumentParser()
    parser.add_argument("--variant", required=True)   # e.g. train/fixed_geo
    args = parser.parse_args()

    for d in sorted((Path("data") / args.variant).iterdir()):
        if not (d / "metadata.json").exists() or (d / "m.npy").exists():
            continue
        p = SimulationParams.from_json(d)
        _, init_m_fn = random_init(jr.split(jnp.array(p.key, dtype=jnp.uint32))[1])
        run(params=p, init_m_fn=init_m_fn, out_path=d / "m.npy")
```

Note `SimulationParams.from_json` already exists and is otherwise unused — it was
written for exactly this. Reconstructing the init function from the key (rather than
parsing the `init` *label*) is what makes `"random"` reproducible; the label alone is
not enough, since a random init needs the key that drew it.

> **Not yet run.** The snippet above is written from a reading of `generate.py` and
> the verified key derivation, but it has never been executed — regenerating needs a
> GPU job, which was out of scope when these notes were written. Treat it as a
> correct-looking starting point, not a tested tool. Check the first sample it writes
> against a surviving `m.npy` before trusting a full run.

**Cost.** Measured from the mtimes of the existing samples: median **9.9 s per
trajectory** at 256×256, mean 22 s (the mean is higher because `relax()` takes longer
from some initial conditions). Actual wall time when they were built: **3.2 h for the
512 train trajectories, 0.9 h for the 128 val**, one GPU each. Budget ~4 GPU-hours to
rebuild both, and 49 GB of disk.

**Bit-exactness.** The keys are deterministic, so the *inputs* reproduce exactly. The
solver output will not be bit-identical on different hardware, a different CUDA/JAX
version, or a different neuralmag release — this is float32 GPU arithmetic through an
adaptive integrator. Expect statistically identical data, not identical bytes. If you
need the numbers in `BENCHMARKS.md` to be exactly reproducible, copy the frames.

## Regenerating the benchmarks

`sp4` and `sp4_xlarge` are the easy case — they are pure functions of the mesh size,
with no random draw at all (`generate_sp4` hardcodes SP4's field and s-state init):

```bash
N=100  NAME=sp4        sbatch scripts/generate.slrm   # writes (100,100); see caveat
N=4000 NAME=sp4_xlarge sbatch scripts/generate.slrm   # ~6 h, 19 GB
```

Caveat on `sp4`: the real SP4 geometry is 100×25 cells, not square, and
`main_sp4` only takes a single `--n` and builds `(n, n)`. Regenerating the
existing `val/sp4` needs `--n` split into two arguments, or a one-off call to
`generate_sp4(n=(100, 25), ...)` directly. It is 1.2 MB — copy it instead.

### Deleted variants

Two benchmark variants were deleted for quota and are **not** currently on disk:

| variant | mesh | field | deleted | still referenced by |
|---|---|---|---|---|
| `large` | (2000, 2000) | in-distribution (random draw) | ~2026-08-28 | `config.BENCHMARK_VARIANTS`, CLAUDE.md table, BENCHMARKS.md, `figures/make_figures.py` Fig 5 |
| `sp4_large` | (3000, 3000) | SP4 | 2026-08-28 | BENCHMARKS.md only |

**`large` currently breaks the eval path.** `config.BENCHMARK_VARIANTS` still lists
it, so `benchmark_path("large")` raises `FileNotFoundError` and every one of
`scripts/eval`, `eval.slrm`, `bench.slrm`, `eval_dt.slrm`, `smoke.slrm` fails at
startup, before touching a GPU. Either regenerate it or drop it from the tuple.

`sp4_large` is easy to bring back — it has no random component:

```bash
N=3000 NAME=sp4_large sbatch scripts/generate.slrm   # ~15 min
```

**`large` cannot be reproduced exactly.** It was made by `main_fixed_geo`
(2000×2000, `generate_sample` → *random* `H_ext` and *random* init), then renamed
from `xlarge` to `large`. Its `metadata.json` — and therefore its key — was deleted
with it, and the seed and sample index that produced it were never recorded anywhere.
A regenerated `large` will be the same *kind* of trajectory (same mesh, same material,
an in-distribution field) but a different draw, so its numbers will not match the
`large` rows in `BENCHMARKS.md`. Those rows are now historical: quote them as such, or
regenerate and re-measure the whole column.

To make a replacement:

```python
generate_sample(key=jr.PRNGKey(<pick one and record it>), n_samples=1, index=1,
                base_dir=Path("data/val/large"), n=(2000, 2000),
                dx=(5e-9, 5e-9, 3e-9), t_tot=1e-9, dt=10e-12)
```

Record the seed in this file when you do.

## Moving machines: what to copy

| path | size | copy? |
|---|---|---|
| `data/` | 66 GB | Copy if you want the recorded numbers to reproduce exactly, or to skip 4 GPU-hours. Otherwise regenerate from `manifest/metadata.zip`. |
| `.nm_cache/` | 4.1 GB | Copy. Rebuildable, but each demag tensor is O(minutes) and it holds the 4000² and 5000² ones. |
| `weights/` | 76 MB | **Copy — gitignored and not regenerable.** `run1/best.eqx` is the 0.0177 headline model. Retraining it is ~35 GPU-hours and will not land on the same weights. |
| `.jax_cache/` | 7.9 GB | Do not copy. Compilation cache, keyed to this GPU and JAX build. |
| `.venv/` | 6.5 GB | Do not copy. `uv sync` rebuilds it; all deps are on PyPI and pinned in `uv.lock`. |
| `wandb/`, `logs/` | small | Optional; run history only. |

Minimum to carry for a working repo with reproducible results: `data/`, `weights/`,
`.nm_cache/`. Minimum to carry to be *able* to reproduce from scratch: just the git
clone (`manifest/metadata.zip` is in it) plus `weights/`.
