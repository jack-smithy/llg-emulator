import logging
import os
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from llg.constants import DT, MATERIAL, MU_0

# importing neuralmag turns off JAX's GPU preallocation; the fragmentation that follows
# makes a 32 x 256^2 training batch miss a 24 GiB slice, so restore whatever was set
_preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
import neuralmag as nm
from neuralmag.backends.jax.demag_field import h_cell
from neuralmag.backends.jax.llg_solver_jax import llg_rhs

if _preallocate is None:
    os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
else:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = _preallocate
nm.set_log_level(logging.WARNING)

NM_CACHE = Path(__file__).resolve().parent.parent / ".nm_cache"
SCALE_T = 1e-9  # neuralmag integrates in ns

Stepper = Callable[[Float[Array, "..."], Float[Array, "..."]], Float[Array, "..."]]


def exchange_field(
    m: Float[Array, "..."], dx: tuple[float, ...], Ms: float, A: float
) -> Float[Array, "..."]:
    """magnum.np's 5-point exchange field in A/m, zero flux through the film's edges."""
    mp = jnp.pad(m, ((1, 1), (1, 1), (0, 0)), mode="edge")
    lap_x = (mp[2:, 1:-1] + mp[:-2, 1:-1] - 2 * m) / dx[0] ** 2
    lap_y = (mp[1:-1, 2:] + mp[1:-1, :-2] - 2 * m) / dx[1] ** 2
    return 2 * A / (MU_0 * Ms) * (lap_x + lap_y)


class DemagField(eqx.Module):
    """neuralmag's FFT demag field of a cell-centred film, in A/m."""

    N: Float[Array, "..."]
    n: tuple[int, int] = eqx.field(static=True)
    Ms: float = eqx.field(static=True)

    def __init__(self, n: tuple[int, int], dx: tuple[float, ...], Ms: float):
        state = nm.State(nm.Mesh(n, dx))
        state.m = nm.VectorFunction(state).fill((0, 0, 1))
        state.material.Ms = Ms
        nm.DemagField(p=20, cache_dir=NM_CACHE).register(state)
        self.N = jnp.asarray(state.N_demag)
        self.n = n
        self.Ms = Ms

    def __call__(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        Ms = jnp.full(self.n + (1,), self.Ms, m.dtype)
        rho = jnp.ones(self.n + (1,), m.dtype)
        return h_cell(self.N, m[:, :, None], Ms, rho)[:, :, 0]


class LLGStepper(eqx.Module):
    """One DT step of the LLG under exchange, demag and an applied field, matched to
    magnum.np: neuralmag's right-hand side through diffrax, as `nm.LLGSolver.step`."""

    demag: DemagField
    dx: tuple[float, ...] = eqx.field(static=True)
    Ms: float = eqx.field(static=True)
    A: float = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, n: tuple[int, int], dx: tuple[float, ...], alpha: float):
        self.demag = DemagField(n, dx, MATERIAL["Ms"])
        self.dx = dx
        self.Ms = MATERIAL["Ms"]
        self.A = MATERIAL["A"]
        self.alpha = alpha

    def rhs(self, m: Float[Array, "..."], h: Float[Array, "..."]) -> Float[Array, "..."]:
        h = h + exchange_field(m, self.dx, self.Ms, self.A) + self.demag(m)
        return llg_rhs(h, m, self.alpha)

    def __call__(self, m: Float[Array, "..."], h: Float[Array, "..."]) -> Float[Array, "..."]:
        h = jnp.broadcast_to(jnp.asarray(h, m.dtype), m.shape)
        solution = dfx.diffeqsolve(
            dfx.ODETerm(lambda t, y, h: SCALE_T * self.rhs(y, h)),
            dfx.Dopri5(),
            t0=0.0,
            t1=DT / SCALE_T,
            dt0=1e-14 / SCALE_T,
            y0=m,
            args=h,
            stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-5),
            max_steps=4096,
        )
        m = solution.ys[-1]
        return m / jnp.linalg.norm(m, axis=-1, keepdims=True)


@lru_cache(maxsize=16)
def llg_solver(
    n: tuple[int, int], dx: tuple[float, ...], alpha: float = MATERIAL["alpha"]
) -> LLGStepper:
    """One stepper per mesh: building the demag tensor is slow, and a new one retraces."""
    return LLGStepper(n, dx, alpha)


@eqx.filter_jit
def rollout(
    step: Stepper, m0: Float[Array, "..."], h: Float[Array, "..."], n_steps: int
) -> Float[Array, "..."]:
    """The n_steps frames after m0 under a constant field (3,) or one per step (n_steps, 3)."""

    def body(m, h):
        m = step(m, h)
        return m, m

    # rematerialised per step, so backprop through an unroll stores only each step's input
    _, frames = jax.lax.scan(jax.checkpoint(body), m0, jnp.broadcast_to(h, (n_steps, 3)))
    return frames


@eqx.filter_jit
def rollout_batch(
    step: Stepper, m0: Float[Array, "..."], h: Float[Array, "..."], n_steps: int
) -> Float[Array, "..."]:
    return jax.vmap(lambda m, h: rollout(step, m, h, n_steps))(m0, h)
