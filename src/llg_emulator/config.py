from pathlib import Path

JAX_CACHE_DIR = ".jax_cache"

DATA_ROOT = Path("data/check")


def dataset_dir(split: str, size: str) -> Path:
    """Path to a dataset split, e.g. dataset_dir("train")."""
    return DATA_ROOT / split
