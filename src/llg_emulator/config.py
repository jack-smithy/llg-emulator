from pathlib import Path

JAX_CACHE_DIR = ".jax_cache"

DATA_ROOT = Path("../micromagnetic-data/data/dynamics")
SIZE = "med"  # or "med" for the bigger dataset
SP4_PATH = DATA_ROOT / SIZE / "sp4" / "sample_0"


def dataset_dir(split: str, size: str = SIZE) -> Path:
    """Path to a dataset split, e.g. dataset_dir("train")."""
    return DATA_ROOT / size / split
