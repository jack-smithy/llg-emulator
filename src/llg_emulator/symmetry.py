"""Exact symmetry group of the LLG dynamics on a square thin-film mesh.

The one-step map m_t -> m_{t+1} conditioned on H_ext is equivariant under the
group that jointly transforms the grid AND the in-plane vector components:
the square-lattice point group D4 (4 rotations x 2 reflections = 8 elements)
times the out-of-plane flip Z2 (mz -> -mz) = 16 elements.

Verified numerically against the neuralmag demag operator (the only non-local
coupling): demag(g.m) == g.demag(m) to float32 noise for every generator.

Applying these to (m0, m1, H) triples gives physically exact extra training
samples — the highest-leverage move for a tiny dataset that must generalise to
an out-of-distribution applied field (the rotations spread the training fields
across all four quadrants, covering SP4's reversing-field angle).

A field is channel-first (3, nx, ny); H is a plain (3,) vector. Every op acts
consistently on both so equivariance holds for the (field, H) pair.
"""

import numpy as np


def _rot90(field):
    """Grid 90deg CCW + in-plane vector rotation (mx,my)->(-my,mx)."""
    g = np.rot90(field, k=1, axes=(-2, -1))
    mx, my, mz = g[..., 0, :, :], g[..., 1, :, :], g[..., 2, :, :]
    return np.stack([-my, mx, mz], axis=-3)


def _flipx(field):
    """Mirror across the x-index: flip grid axis + reverse mx."""
    g = np.flip(field, axis=-2)
    mx, my, mz = g[..., 0, :, :], g[..., 1, :, :], g[..., 2, :, :]
    return np.stack([-mx, my, mz], axis=-3)


def _zflip(field):
    mx, my, mz = field[..., 0, :, :], field[..., 1, :, :], field[..., 2, :, :]
    return np.stack([mx, my, -mz], axis=-3)


def _rotH(H):
    return np.array([-H[1], H[0], H[2]], dtype=H.dtype)


def _flipxH(H):
    return np.array([-H[0], H[1], H[2]], dtype=H.dtype)


def _zflipH(H):
    return np.array([H[0], H[1], -H[2]], dtype=H.dtype)


def _decode(g: int):
    """Group element index in [0,16) -> (n_rot in 0..3, do_flipx, do_zflip)."""
    return g % 4, (g // 4) % 2, (g // 8) % 2


NUM_ELEMENTS = 16


def apply_field(field, g: int):
    """Apply group element g to a channel-first field (..., 3, nx, ny)."""
    r, fx, zf = _decode(g)
    for _ in range(r):
        field = _rot90(field)
    if fx:
        field = _flipx(field)
    if zf:
        field = _zflip(field)
    return np.ascontiguousarray(field)


def apply_H(H, g: int):
    """Apply the same group element to an applied field vector (3,)."""
    r, fx, zf = _decode(g)
    H = np.asarray(H)
    for _ in range(r):
        H = _rotH(H)
    if fx:
        H = _flipxH(H)
    if zf:
        H = _zflipH(H)
    return H.astype(np.float32)


def _demo():
    """Self-check: group has 16 distinct elements and demag is equivariant."""
    import jax
    import jax.numpy as jnp

    from llg_emulator.physics import DemagField

    dem = DemagField((31, 31), (5e-9, 5e-9, 3e-9), Ms=1.0, p=20)
    m = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (3, 32, 32)))
    m = m / np.linalg.norm(m, axis=0, keepdims=True)

    for g in range(NUM_ELEMENTS):
        lhs = np.asarray(dem(jnp.asarray(apply_field(m, g))))
        rhs = apply_field(np.asarray(dem(jnp.asarray(m))), g)
        rel = np.abs(lhs - rhs).max() / (np.abs(rhs).max() + 1e-12)
        assert rel < 1e-4, f"g={g} not equivariant: {rel}"

    # distinct: the 16 elements give 16 distinct results on a generic field
    outs = {apply_field(m, g).tobytes() for g in range(NUM_ELEMENTS)}
    assert len(outs) == NUM_ELEMENTS, f"group collapsed to {len(outs)}"
    print("symmetry self-check ok: 16 distinct, demag-equivariant elements")


if __name__ == "__main__":
    _demo()
