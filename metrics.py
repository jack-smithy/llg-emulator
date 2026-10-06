"""Metrics of magnetisation fields (..., nx, ny, 3), reduced over the film to (...)."""

import jax.numpy as jnp
from jaxtyping import Array, Float

Field = Float[Array, "..."]


def mse(pred: Field, truth: Field) -> Field:
    return ((pred - truth) ** 2).mean(axis=(-3, -2, -1))


def vrmse(pred: Field, truth: Field, eps: float = 1e-7) -> Field:
    """the_well's VRMSE: per component, normalised by the truth's spatial variance."""
    err = ((pred - truth) ** 2).mean(axis=(-3, -2))
    var = truth.var(axis=(-3, -2), ddof=1)
    return jnp.sqrt(err / (var + eps)).mean(axis=-1)


def cosine_similarity(pred: Field, truth: Field) -> Field:
    return (pred * truth).sum(axis=-1).mean(axis=(-2, -1))


METRICS = {"mse": mse, "vrmse": vrmse, "cos": cosine_similarity}


def loop_metrics(h_mt: Field, m: Field) -> dict[str, float]:
    """Coercive field and remanence of each branch of a +H -> -H -> +H loop, and its area."""
    half = len(h_mt) // 2
    out = {}
    for name, branch in (("down", slice(0, half)), ("up", slice(half, None))):
        h, mb = h_mt[branch], m[branch]
        out[f"hc_{name}"] = zero_crossing(h, mb)
        out[f"mr_{name}"] = float(mb[jnp.argmin(jnp.abs(h))])
    out["area"] = float(jnp.abs(jnp.trapezoid(m, h_mt)))
    return out


def zero_crossing(x: Field, y: Field) -> float:
    """x where y first changes sign, linearly interpolated; nan if it never does."""
    i = jnp.flatnonzero(jnp.diff(jnp.sign(y)))
    if len(i) == 0:
        return float("nan")
    i = i[0]
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))
