"""Project constants. Imports only os/pathlib (no jax) so this module can be
imported before CUDA_VISIBLE_DEVICES is set."""

from pathlib import Path

JAX_CACHE_DIR = ".jax_cache"

DATA_ROOT = Path("data")
SIZE = "small"  # or "med" for the bigger dataset
SP4_PATH = DATA_ROOT / "sp4" / "sample_0"

RESULTS_DIR = Path("results")


def dataset_dir(split: str, size: str = SIZE) -> Path:
    """Path to a dataset split, e.g. dataset_dir("train")."""
    return DATA_ROOT / size / split
