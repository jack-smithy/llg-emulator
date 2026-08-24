"""Demag (magnetostatic) field calculator built on neuralmag.

The demag field is the long-range, geometry-coupled term in the LLG effective
field: solving it cheaply and exactly here lets the neural surrogate spend its
capacity on the short-range interactions (exchange, anisotropy) instead of
re-learning a global convolution.

neuralmag computes the field as an FFT convolution of the magnetization with a
precomputed demag tensor `N` that depends only on the mesh geometry (cell count
+ cell size). We build that tensor once via neuralmag's `State`/`DemagField`
machinery at construction time, store it as a (non-trainable) buffer, and apply
neuralmag's `h_cell` convolution on every call. The module is therefore a pure,
jit/vmap-friendly Equinox module with no learnable parameters.
"""

import logging as _logging
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import neuralmag as nm
from jaxtyping import Array
from neuralmag.backends.jax.demag_field import h_cell

# neuralmag logs its setup at INFO on every State build (one per model
# construction); keep the training/eval output readable.
nm.set_log_level(_logging.WARNING)

# The mesh geometry is fixed across runs, so the demag tensor is built once and
# reused (keyed on n/dx/pbc/p by neuralmag). Shared with src/datagen.
NM_CACHE = Path(__file__).resolve().parents[2] / ".nm_cache"


def _build_demag_tensor(n, dx, Ms: float, p: int) -> Array:
    """Precompute neuralmag's demag tensor `N` for a given mesh geometry.

    Returns the (3, 3, *) rfft tensor as a single jax array. Triggers
    neuralmag's one-time code generation, so call this sparingly.
    """
    mesh = nm.Mesh(tuple(n), tuple(dx))
    state = nm.State(mesh)
    state.m = nm.VectorCellFunction(state).fill((0, 0, 1))
    state.material.Ms = Ms
    nm.DemagField(p=p, cache_dir=NM_CACHE).register(state)
    # State stores N_demag as a stacked (3, 3, nx_pad, ny_rfft, nz) array.
    return jnp.asarray(state.N_demag)


class DemagField(eqx.Module):
    """Computes the demag field of a magnetization field via neuralmag.

    Call signature mirrors the model's: a single channel-first sample
    `m` of shape (3, A, B) (unit vectors) maps to the demag field of the same
    shape. Vmap externally for batches.

    The output is nondimensionalized by `Ms` (so it matches the H_ext / Ms
    convention used for the applied-field input channels); pass
    `nondim=False` for the raw A/m field.
    """

    N: Array  # (3, 3, *) demag tensor (rfft); buffer, not trained
    Ms: float = eqx.field(static=True)
    n: tuple = eqx.field(static=True)  # (nx, ny, nz) mesh cell counts
    nondim: bool = eqx.field(static=True)

    def __init__(
        self,
        n,
        dx,
        Ms: float,
        *,
        p: int = 20,
        nondim: bool = True,
    ):
        self.n = tuple(int(x) for x in n)
        self.Ms = float(Ms)
        self.nondim = nondim
        self.N = _build_demag_tensor(self.n, dx, self.Ms, p)

    def __call__(self, m: Array) -> Array:
        # (3, A, B) -> (A, B, nz=1, 3) cell-vector layout neuralmag expects.
        nx, ny, nz = self.n
        m_cell = jnp.moveaxis(m, 0, -1).reshape(nx, ny, nz, 3)

        Ms_field = jnp.full((nx, ny, nz), self.Ms, dtype=m.dtype)
        rho = jnp.ones((nx, ny, nz), dtype=m.dtype)

        h = h_cell(self.N, m_cell, Ms_field, rho)  # (nx, ny, nz, 3) in A/m
        if self.nondim:
            h = h / self.Ms

        return jnp.moveaxis(h.reshape(nx, ny, 3), -1, 0)  # (3, A, B)
