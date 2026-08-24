import json
from pathlib import Path

import equinox as eqx
import numpy as np
from pyevtk.hl import gridToVTK

from llg_emulator.model import LLGEmulator, ModelConfig


def write_vtr(field, filename, n, dx):
    # save result
    x = np.arange(0, (0.1 + n[0]) * dx[0], dx[0], dtype="float64")
    y = np.arange(0, (0.1 + n[1]) * dx[1], dx[1], dtype="float64")
    z = np.arange(0, (0.1 + n[2]) * dx[2], dx[2], dtype="float64")

    # scalar data
    if len(field.shape) == 3:
        gridToVTK(filename, x, y, z, cellData={"f": field.copy()})
    else:
        gridToVTK(
            filename,
            x,
            y,
            z,
            cellData={
                "f": (
                    field[:, :, :, 0].copy(),
                    field[:, :, :, 1].copy(),
                    field[:, :, :, 2].copy(),
                )
            },
        )


def save_model(model: LLGEmulator, config: ModelConfig, path: Path, tag: str):
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / f"{tag}.eqx", model)
    meta = {"hidden_channels": config.hidden_channels, "num_blocks": config.num_blocks}
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta))


def load_model(path, demag, key, tag="best") -> LLGEmulator:
    meta = json.loads((path / "metadata.json").read_text())
    config = ModelConfig(
        hidden_channels=meta["hidden_channels"],
        num_blocks=meta["num_blocks"],
    )
    skeleton = LLGEmulator(config=config, demag=demag, key=key)
    return eqx.tree_deserialise_leaves(path / f"{tag}.eqx", skeleton)
