from pathlib import Path

JAX_CACHE_DIR = ".jax_cache"

DATA_ROOT = Path("data")

# Held-out benchmarks, kept out of the val one-step loss. Each is a single
# trajectory on a mesh other than the training mesh, so none of them could share
# a batch with the val set anyway; the point is to measure transfer, not to fit.
#   sp4   -- 100x25 cells, out-of-distribution applied field
#   large -- 2000x2000 cells, 61x the training area: the domain-size check
SP4_VARIANT = "sp4"
BENCHMARK_VARIANTS = ("sp4", "large")


def variant_dirs(split: str, root: Path = DATA_ROOT, exclude: tuple = ()) -> list[Path]:
    """Variant dirs holding samples, e.g. data/train/fixed_geo.

    Every subdirectory of `data/<split>/` is a variant; new ones are picked up
    with no code change.
    """
    base = root / split
    dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name not in exclude)
    if not dirs:
        raise FileNotFoundError(f"no variant dirs under {base}")
    return dirs


def benchmark_path(variant: str, root: Path = DATA_ROOT) -> Path:
    """The single sample dir of a benchmark variant.

    Unpacking fails loudly if a benchmark ever gains a second trajectory — the
    eval paths report one number per variant and would silently ignore the rest.
    """
    (path,) = sorted(p for p in (root / "val" / variant).iterdir() if p.is_dir())
    return path


def sp4_path(root: Path = DATA_ROOT) -> Path:
    return benchmark_path(SP4_VARIANT, root)
