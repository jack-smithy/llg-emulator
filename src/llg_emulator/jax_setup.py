"""Call configure_jax() once, after CUDA_VISIBLE_DEVICES is set, before use."""

import jax

from llg_emulator.config import JAX_CACHE_DIR


def configure_jax() -> None:
    jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
