"""Call configure_jax() once before JAX is used to persist the JIT cache."""

import jax

from llg_emulator.config import JAX_CACHE_DIR


def configure_jax() -> None:
    jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_num_cpu_devices", 8)
