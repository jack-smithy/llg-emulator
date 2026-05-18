from datetime import datetime
from pathlib import Path

import equinox as eqx
from jaxtyping import PRNGKeyArray

from llg_emulator.config import RESULTS_DIR
from llg_emulator.model import LLGEmulator


def load_model(key: PRNGKeyArray, weights_path) -> LLGEmulator:
    """Rebuild an LLGEmulator (default architecture) and load its weights."""
    model = LLGEmulator(key=key)
    return eqx.tree_deserialise_leaves(weights_path, model)


def make_run_dir(root: Path = RESULTS_DIR) -> Path:
    """Fresh timestamped run dir (with checkpoint subdirs) so runs never
    overwrite each other."""
    run_dir = Path(root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (run_dir / "checkpoints" / "trjs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "weights").mkdir(parents=True, exist_ok=True)
    return run_dir
