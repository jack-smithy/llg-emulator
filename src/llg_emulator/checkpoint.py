from datetime import datetime
from pathlib import Path

import equinox as eqx
from jaxtyping import PRNGKeyArray

from llg_emulator.config import RESULTS_DIR
from llg_emulator.experiment import ModelConfig, build_activation
from llg_emulator.model import LLGEmulator


def load_model(
    key: PRNGKeyArray,
    weights_path,
    model_config: ModelConfig | None = None,
) -> LLGEmulator:
    """Rebuild an LLGEmulator from its architecture config and load weights.

    Pass the run's ModelConfig (from TrainConfig.from_run_dir) so checkpoints
    trained with a non-default architecture reload correctly. Defaults to
    ModelConfig() (the historical default architecture).
    """
    cfg = model_config or ModelConfig()
    model = LLGEmulator(
        hidden_channels=cfg.hidden_channels,
        num_blocks=cfg.num_blocks,
        activation=build_activation(cfg.activation),
        mesh_n=cfg.mesh_n,
        mesh_dx=cfg.mesh_dx,
        demag_p=cfg.demag_p,
        key=key,
    )
    return eqx.tree_deserialise_leaves(weights_path, model)


def make_run_dir(root: Path = RESULTS_DIR) -> Path:
    """Fresh timestamped run dir (with checkpoint subdirs) so runs never
    overwrite each other."""
    run_dir = Path(root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (run_dir / "checkpoints" / "trjs").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints" / "weights").mkdir(parents=True, exist_ok=True)
    return run_dir
