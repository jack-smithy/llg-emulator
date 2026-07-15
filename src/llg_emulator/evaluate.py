from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.random as jr

from llg_emulator.config import JAX_CACHE_DIR, dataset_dir
from llg_emulator.data import LLGStepperSource
from llg_emulator.io import load_model
from llg_emulator.physics import DemagField
from llg_emulator.training import count_parameters

jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--size", type=str, default="small")
    parser.add_argument("--model-path", type=str, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    seed = args.seed

    model_path = Path(args.model_path)

    device = jax.devices()[0]

    val_shards = dataset_dir("val", args.size)
    val_dataset = LLGStepperSource(val_shards, num_shards=None)

    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)
    demag = DemagField((256, 256, 1), (5e-9, 5e-9, 3e-9), Ms=1.0, p=20)
    model = load_model(path=model_path, demag=demag, tag="weights", key=subkey)

    print(count_parameters(model), len(val_dataset), device)


if __name__ == "__main__":
    main()
