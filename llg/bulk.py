"""Magnetostatics and energy minimisation of a soft-magnetic cuboid on a regular 3D mesh:
the NiFe shield of the TMR study (D. Suess, "3D TMR sensors ... shield optimization", 2026).

Three models of the same body, all with the exact demag field of the mesh:

- micromagnetics, |m| = 1, exchange constant A, relaxed to a local energy minimum;
- the coarse-grained domain solver of the study, which is the same thing with
  A_eff = sigma dx / 2: its wall penalty sigma / (mu0 Ms dx) sum_j (m_j - m_i) is the
  discrete exchange field 2 A / (mu0 Ms dx^2) sum_j (m_j - m_i) with that A;
- the soft-magnet limit of the study's Maxwell solver (chi -> infinity with saturation):
  the convex problem of minimal demag + Zeeman energy over |m| <= 1, whose minimum is unique.

Fields are in units of Ms unless named h_*_am (A/m). Lengths in metres.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from llg.constants import MU_0
from llg.physics import NM_CACHE, h_cell, nm

Field = Float[Array, "nx ny nz 3"]


class Body(eqx.Module):
    """A cuboid of cells n with sizes dx, centred at the origin."""

    N: Float[Array, "..."]
    n: tuple[int, int, int] = eqx.field(static=True)
    dx: tuple[float, float, float] = eqx.field(static=True)
    Ms: float = eqx.field(static=True)

    def __init__(self, n: tuple[int, int, int], dx: tuple[float, float, float], Ms: float):
        state = nm.State(nm.Mesh(n, dx))
        state.m = nm.VectorFunction(state).fill((0, 0, 1))
        state.material.Ms = Ms
        nm.DemagField(p=20, cache_dir=NM_CACHE).register(state)
        self.N = jnp.asarray(state.N_demag)
        self.n = n
        self.dx = dx
        self.Ms = Ms

    def demag(self, m: Field) -> Field:
        ones = jnp.ones(self.n, m.dtype)
        return h_cell(self.N.astype(m.dtype), m, self.Ms * ones, ones) / self.Ms

    def exchange(self, m: Field, A: float) -> Field:
        """6-point exchange field, zero flux through the surface."""
        mp = jnp.pad(m, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
        core = mp[1:-1, 1:-1, 1:-1]
        lap = (
            (mp[2:, 1:-1, 1:-1] + mp[:-2, 1:-1, 1:-1] - 2 * core) / self.dx[0] ** 2
            + (mp[1:-1, 2:, 1:-1] + mp[1:-1, :-2, 1:-1] - 2 * core) / self.dx[1] ** 2
            + (mp[1:-1, 1:-1, 2:] + mp[1:-1, 1:-1, :-2] - 2 * core) / self.dx[2] ** 2
        )
        return 2 * A / (MU_0 * self.Ms**2) * lap

    def centres(self) -> Float[Array, "nx ny nz 3"]:
        axes = [(jnp.arange(n) + 0.5 - n / 2) * d for n, d in zip(self.n, self.dx)]
        return jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)


def normalise(m: Field) -> Field:
    return m / jnp.linalg.norm(m, axis=-1, keepdims=True)


def minimise_sphere(field, m0: Field, tol: float = 1e-5, max_iter: int = 50_000):
    """A local minimum of the energy with effective field `field(m)` (units of Ms) over
    |m| = 1: steepest descent along the torque with alternating Barzilai-Borwein steps
    (Exl et al., J. Appl. Phys. 115, 17D118 (2014)). Returns (m, iterations, max torque)."""

    def torque(m):
        h = field(m)
        return h - (m * h).sum(-1, keepdims=True) * m

    def step(carry):
        i, m, t, tau = carry
        m_new = normalise(m + tau * t)
        t_new = torque(m_new)
        s, y = m_new - m, t - t_new  # y: change of the gradient, which is -torque
        sy = (s * y).sum()
        tau_new = jnp.where(i % 2 == 0, (s * s).sum() / sy, sy / (y * y).sum())
        tau_new = jnp.where(jnp.isfinite(tau_new) & (tau_new > 0), tau_new, tau)
        return i + 1, m_new, t_new, jnp.clip(tau_new, 1e-6, 1e4)

    def not_done(carry):
        i, _, t, _ = carry
        return (i < max_iter) & (jnp.abs(t).max() > tol)

    i, m, t, _ = jax.lax.while_loop(not_done, step, (0, m0, torque(m0), jnp.asarray(1e-2, m0.dtype)))
    return m, i, jnp.abs(t).max()


def minimise_ball(field, m0: Field, tol: float = 1e-7, max_iter: int = 50_000):
    """The minimum over |m| <= 1 of a convex energy whose negative gradient is field(m)
    (Lipschitz constant <= 1, as for the demag operator in units of Ms): FISTA with
    adaptive restart. Returns (m, iterations, max change of the last step)."""

    def project(m):
        return m / jnp.maximum(1.0, jnp.linalg.norm(m, axis=-1, keepdims=True))

    def step(carry):
        i, m, y, t, _ = carry
        m_new = project(y + field(y))
        t_new = (1 + jnp.sqrt(1 + 4 * t**2)) / 2
        restart = ((y - m_new) * (m_new - m)).sum() > 0
        t_new = jnp.where(restart, 1.0, t_new)
        momentum = jnp.where(restart, 0.0, (t - 1) / t_new)
        return i + 1, m_new, m_new + momentum * (m_new - m), t_new, jnp.abs(m_new - m).max()

    def not_done(carry):
        i, *_, change = carry
        return (i < max_iter) & (change > tol)

    one = jnp.asarray(1.0, m0.dtype)
    i, m, *_, change = jax.lax.while_loop(not_done, step, (0, m0, m0, one, one))
    return m, i, change


def _log_plus(v, R, rest):
    """log(v + R) with R = sqrt(v^2 + rest), without cancellation for v < 0."""
    tiny = jnp.finfo(R.dtype).tiny
    return jnp.where(v >= 0, jnp.log(v + R), jnp.log(jnp.maximum(rest, tiny) / (R - v)))


def _face_field(r, centre, half, axis: int, sigma):
    """H / sigma * 4 pi at the point r of rectangles normal to `axis` (centres and half
    sides (F, 3)), summed with charge densities sigma (F,)."""
    b, d = [i for i in range(3) if i != axis]
    w = r[axis] - centre[:, axis]
    w = jnp.where(w == 0, jnp.finfo(w.dtype).tiny, w)  # in the face's plane: terms cancel
    h = jnp.zeros((3,) + w.shape, w.dtype)
    for k, sb in enumerate((-1, 1)):
        u = r[b] - (centre[:, b] + sb * half[:, b])
        for l, sd in enumerate((-1, 1)):
            v = r[d] - (centre[:, d] + sd * half[:, d])
            R = jnp.sqrt(u**2 + v**2 + w**2)
            sign = (-1) ** (k + l)
            h = h.at[axis].add(sign * jnp.arctan(u * v / (w * R)))
            h = h.at[b].add(-sign * _log_plus(v, R, u**2 + w**2))
            h = h.at[d].add(-sign * _log_plus(u, R, v**2 + w**2))
    return (h * sigma).sum(axis=1) / (4 * jnp.pi)


def stray_field(body: Body, m: Field, points: Float[Array, "p 3"]) -> Float[Array, "p 3"]:
    """The demag field in A/m at arbitrary points, exactly for cells of uniform m: the
    field of each cell's six charged faces. No air box, no interpolation."""
    centres = body.centres().reshape(-1, 3)
    M = body.Ms * m.reshape(-1, 3)
    half = jnp.broadcast_to(jnp.asarray(body.dx, m.dtype) / 2, centres.shape)

    def at(r):
        h = jnp.zeros(3, m.dtype)
        for axis in range(3):
            offset = jnp.zeros(3, m.dtype).at[axis].set(body.dx[axis] / 2)
            h += _face_field(r, centres + offset, half, axis, M[:, axis])
            h += _face_field(r, centres - offset, half, axis, -M[:, axis])
        return h

    return jax.lax.map(at, jnp.asarray(points, m.dtype))
