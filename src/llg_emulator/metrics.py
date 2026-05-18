import jax.numpy as jnp
from jaxtyping import Array


def nRMSE(pred: Array, ref: Array) -> Array:
    return jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)


def correlation(pred: Array, ref: Array) -> Array:
    pred_n = pred / jnp.linalg.norm(pred)
    ref_n = ref / jnp.linalg.norm(ref)
    return jnp.dot(pred_n.flatten(), ref_n.flatten())
