import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from einops import rearrange
from jaxtyping import PyTree
from torch.utils.data import default_collate
from tqdm import tqdm

# `batch["constant_scalars"]` in the order the generator writes them
# (datagen.generate_varied_field.create_split); main.py checks the dataset agrees.
SCALARS = ("Ms", "A", "alpha", "Hx", "Hy", "Hz")

MU_0 = 4e-7 * math.pi
# the conditioning is divided by this so it is O(1): ~ the largest |H| / Ms in the data
# (50 mT / mu0 / 800 kA/m = 0.0497)
H_SCALE = 0.05
# `edge_features` saturate this far (um) from a film edge: past it the edge no
# longer shapes the local dynamics (the edge charges' own field comes in exactly
# through the demag channels)
EDGE_SCALE_UM = 0.32
COORDS = ("absolute", "edge", "none")


def grid_coords(grid):
    """The cell-centre coordinates in um as two channels.

    grid: (B, Lx, Ly, 2), `space_grid` as WellDataset serves it, i.e. the
    file's own (x, y) cell centres in metres. Taking it from the sample puts
    every geometry on the training set's scale — but the values leave the
    training range on any film larger than the training one.
    returns: (B, 2, Lx, Ly)
    """
    return rearrange(grid, "B Lx Ly D -> B D Lx Ly") * 1e6


def edge_features(grid):
    """Signed distance to the film edges per axis, saturating: bounded coordinates.

    grid: (B, Lx, Ly, 2) cell centres in metres. returns: (B, 2, Lx, Ly), per axis
    `(min(d_lo, s) - min(d_hi, s)) / s` with d the distance from the cell centre
    to the physical film edge and `s = EDGE_SCALE_UM`: about -1 at the low edge,
    +1 at the high edge, 0 in the interior. This is the translation-invariant
    part of the coordinates' boundary information; unlike `grid_coords` its
    values stay in [-1, 1] whatever the film size.
    """
    coords = grid_coords(grid)
    half = 0.5 * (coords[:, :, 1:2, 1:2] - coords[:, :, :1, :1])  # half a cell
    lo = coords.min(axis=(2, 3), keepdims=True) - half
    hi = coords.max(axis=(2, 3), keepdims=True) + half
    s = EDGE_SCALE_UM
    return (jnp.minimum(coords - lo, s) - jnp.minimum(hi - coords, s)) / s


def spatial_features(grid, coords):
    """The per-cell geometry channels for `ModelConfig.coords`, or None."""
    if coords == "absolute":
        return grid_coords(grid)
    if coords == "edge":
        return edge_features(grid)
    if coords == "none":
        return None
    raise ValueError(f"unknown coords {coords!r}, expected one of {COORDS}")


@eqx.filter_jit
def model_inputs(m, coords, demag=None, solver=None, cond=None):
    """Assemble the network input: [S(m)] + m [+ its demag field] [+ geometry].

    m: (B, 3, Lx, Ly); coords: (B, 2, Lx, Ly) from `spatial_features`, or None
    for a model trained without them; demag: a `physics.DemagField` for this
    mesh (nondimensional, Ms=1) or None; solver: a `physics.CoarseLLG` (neuralmag) for this
    mesh or None, with cond (B, 2 [+1]) from `conditioning` for its field.
    returns: (B, [3 +] 3 [+3] [+2], Lx, Ly). The demag field is the only
    long-range, geometry-coupled term in the LLG effective field; feeding it
    exactly means the network only has to learn the short-range physics, which
    transfers across film sizes. With a solver its one-frame prediction S(m)
    comes first, so it is what `model.ResidualEmulator` adds the network's
    output to: the network is then a learned correction to micromagnetics on
    this mesh.
    """
    parts = []
    if solver is not None:
        h = jnp.pad(cond[:, :2] * H_SCALE, ((0, 0), (0, 1)))  # [Hx, Hy, 0] / Ms
        parts.append(jax.vmap(solver)(m, h))
    parts.append(m)
    if demag is not None:
        parts.append(jax.vmap(demag)(m))
    if coords is not None:
        parts.append(coords)
    return jnp.concatenate(parts, axis=1)


def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data samples into a batch.
    default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
    """
    return jtu.tree_map(np.asarray, default_collate(batch))


def conditioning(scalars, grid=None):
    """`constant_scalars` (B, 6) -> `cond` (B, 2): `[Hx, Hy] / Ms / H_SCALE`, with
    `log(delta / l_ex)` appended, (B, 3), when the sample's `space_grid` is given.

    Hz is always zero in the data and Ms is constant, so the field components
    are the whole per-sample variation, nondimensionalised and scaled to O(1).
    `delta / l_ex` -- the cell size over the exchange length
    `sqrt(2A / (mu0 Ms^2))`, ~0.88 for the native 5 nm permalloy cells -- is the
    one length ratio of the discretised problem, i.e. the coordinates' spacing
    without their absolute offset (`ModelConfig.cell_size_cond`); it enters as a
    log because it is a scale parameter spanning decades.
    """
    h = scalars[:, 3:5] / scalars[:, :1] / H_SCALE
    if grid is None:
        return h
    delta = grid[:, 1, 0, 0] - grid[:, 0, 0, 0]  # x-spacing; in-plane cells are square
    l_ex = jnp.sqrt(2 * scalars[:, 1] / (MU_0 * scalars[:, 0] ** 2))
    return jnp.concatenate([h, jnp.log(delta / l_ex)[:, None]], axis=1)


def resampled_cells(res, factor):
    """`downsample`'s mesh for a film of `res` cells: the same extent covered
    by `round(res / factor)` cells, so any factor >= 1 works — the realized
    factor is `res / round(res / factor)` per axis."""
    if factor < 1:
        raise ValueError(f"pool factor {factor} < 1: finer than the simulation cell")
    return tuple(max(1, round(r / factor)) for r in res)


def resampled_mesh(res, dx, factor):
    """`downsample`'s mesh as (cells, 3d cell size): the same film covered by
    coarser cells, the thickness `dx[2]` unchanged. What `physics.demag_cache`
    takes for a resampled film."""
    n = resampled_cells(res, factor)
    return n, tuple(r * d / c for r, d, c in zip(res, dx, n)) + (dx[2],)


@functools.cache
def cell_average_weights(n_fine, n_coarse):
    """(n_coarse, n_fine) weights of the cell average between two uniform 1D grids
    over the same extent: the overlap of each coarse cell with each fine cell, as
    a fraction of the coarse cell, so every row sums to one (conservative
    remapping; a plain block mean at integer factors)."""
    fine = np.linspace(0.0, 1.0, n_fine + 1)
    coarse = np.linspace(0.0, 1.0, n_coarse + 1)
    lo = np.maximum(coarse[:-1, None], fine[None, :-1])
    hi = np.minimum(coarse[1:, None], fine[None, 1:])
    return (np.clip(hi - lo, 0.0, None) * n_coarse).astype(np.float32)


def downsample(m, grid, factor: float):
    """Coarse-grain magnetization and grid onto a mesh of `factor`-larger cells.

    m: (B, 3, Lx, Ly) unit vectors; grid: (B', Lx, Ly, 2) cell centres in
    metres (the two leading dims are independent). `factor` is any real >= 1
    (never below the simulation cell): the film's extent is preserved and the
    mesh becomes `resampled_cells` per axis. Each coarse cell's m is the
    average of the fine m over its area (`cell_average_weights`) -- the
    finite-volume definition of a coarse cell's magnetization, which composes
    exactly across scales. The average leaves the unit sphere, so it is
    renormalised to stay on the manifold the model assumes -- near domain walls
    that is lossy, which is inherent to emulating at a larger cell size. The
    grid is rebuilt analytically: uniform centres of the coarse cells over the
    same film.
    returns: (m, grid) on the coarse mesh.
    """
    if factor == 1:
        return m, grid
    nx, ny = resampled_cells(m.shape[-2:], factor)
    wx = cell_average_weights(m.shape[-2], nx)
    wy = cell_average_weights(m.shape[-1], ny)
    m = jnp.einsum("ij,...jk,lk->...il", wx, jnp.asarray(m), wy)
    m = m / jnp.linalg.norm(m, axis=-3, keepdims=True)

    # per axis: film edge and extent from the fine centres, then coarse centres
    lo = grid[:, :1, :1] - 0.5 * (grid[:, 1:2, 1:2] - grid[:, :1, :1])  # (B',1,1,2)
    span = grid[:, -1:, -1:] - grid[:, :1, :1] + (grid[:, 1:2, 1:2] - grid[:, :1, :1])
    fx = (jnp.arange(nx) + 0.5)[:, None, None] * jnp.array([1.0, 0.0])
    fy = (jnp.arange(ny) + 0.5)[None, :, None] * jnp.array([0.0, 1.0])
    grid = lo + span / jnp.array([nx, ny]) * (fx + fy)
    return m, grid


def prepare_batch(
    batch, demag=None, pool=1, coords="absolute", cell_size_cond=False, solver=None
):
    """One-step batch -> `(m0, m1, cond)` in the model's layout.

    Takes numpy or torch leaves: torch tensors cross the DataLoader worker
    boundary through shared memory, which is much cheaper than pickling numpy
    arrays through a pipe, and `np.asarray` on a CPU tensor is a free view.

    `pool` > 1 coarse-grains the batch onto the coarser mesh first; the caller's
    `demag` must be built for that mesh. `coords` and `cell_size_cond` must
    match the model's `ModelConfig`.

    m0: (B, 3 [+3] [+2], Lx, Ly), the frame as `model_inputs` assembles it;
    m1: (B, 3, Lx, Ly); cond: (B, 2 [+1]).
    """
    batch = jtu.tree_map(np.asarray, batch)
    m0 = rearrange(batch["input_fields"], "B 1 Lx Ly F -> B F Lx Ly")
    m1 = rearrange(batch["output_fields"], "B 1 Lx Ly F -> B F Lx Ly")
    m0, grid = downsample(m0, batch["space_grid"], pool)
    m1, _ = downsample(m1, batch["space_grid"], pool)
    c = conditioning(batch["constant_scalars"], grid if cell_size_cond else None)
    feats = spatial_features(grid, coords)

    return model_inputs(m0, feats, demag, solver, c), m1, c


def prepare_unrolled(batch, pool=1, coords="absolute", cell_size_cond=False):
    """K-step batch (`n_steps_output=K`) -> `(m0, feats, targets, cond)` for
    `unrolled_loss_fn`: the raw first frame, since each step's demag input is
    recomputed from the model's own previous prediction.

    m0: (B, 3, Lx, Ly); feats: (B, 2, Lx, Ly) or None; targets: (B, K, 3, Lx, Ly);
    cond: (B, 2 [+1]).
    """
    batch = jtu.tree_map(np.asarray, batch)
    m0 = rearrange(batch["input_fields"], "B 1 Lx Ly F -> B F Lx Ly")
    b, k = batch["output_fields"].shape[:2]
    mt = rearrange(batch["output_fields"], "B K Lx Ly F -> (B K) F Lx Ly")
    m0, grid = downsample(m0, batch["space_grid"], pool)
    mt, _ = downsample(mt, batch["space_grid"][:1], pool)
    targets = rearrange(mt, "(B K) F Lx Ly -> B K F Lx Ly", B=b, K=k)
    c = conditioning(batch["constant_scalars"], grid if cell_size_cond else None)
    return m0, spatial_features(grid, coords), targets, c


@eqx.filter_jit
def predict(model, m0, cond):
    """One step for a batch: `prepare_batch`'s (m0, cond) -> (B, 3, Lx, Ly)."""
    return jax.vmap(model)(m0, cond)


@eqx.filter_jit
def rollout(
    model, x, cond, grid, n_steps: int, demag=None, coords="absolute", solver=None
):
    """Autoregressively predict n_steps frames from one frame.

    x: (B, 1, Lx, Ly, F), as WellDataset serves input_fields.
    cond: (B, 2 [+1]), as `conditioning` returns it for this model.
    grid: (B, Lx, Ly, 2), as WellDataset serves space_grid.
    demag: a `physics.DemagField` for this mesh (Ms=1) or None; a model
    trained with demag inputs recomputes them from its own prediction at
    every step. `coords` must match the model's.
    returns: (B, n_steps, Lx, Ly, F), same layout as output_fields.
    """
    m0 = rearrange(x, "B 1 Lx Ly F -> B F Lx Ly")
    feats = spatial_features(grid, coords)

    def step(m, _):
        m = jax.vmap(model)(model_inputs(m, feats, demag, solver, cond), cond)
        return m, m

    _, frames = jax.lax.scan(step, m0, None, length=n_steps)
    return rearrange(frames, "T B F Lx Ly -> B T Lx Ly F")


def one_step_preds(
    model,
    loader,
    demag=None,
    pool=1,
    coords="absolute",
    cell_size_cond=False,
    solver=None,
):
    """Teacher-forced one-step predictions over a whole loader.

    returns: (pred, truth), both (N, 1, Lx, Ly, F) numpy -- the layout the
    well's metrics reduce over (on the pooled mesh when `pool` > 1).
    """
    preds, truths = [], []
    for batch in tqdm(loader, mininterval=10):
        m0, m1, cond = prepare_batch(batch, demag, pool, coords, cell_size_cond, solver)
        preds.append(np.asarray(predict(model, m0, cond)))
        truths.append(m1)

    pattern = "B F Lx Ly -> B 1 Lx Ly F"
    return (
        rearrange(np.concatenate(preds), pattern),
        rearrange(np.concatenate(truths), pattern),
    )


def relative_norm_error(m):
    norm = np.linalg.norm(m, axis=-1)
    return norm.mean((1, 2))


def device_info():
    print(jax.devices())


@eqx.filter_jit
def loss_fn(model, m0, m1, cond):
    return jnp.mean(jnp.square(jax.vmap(model)(m0, cond) - m1))


@eqx.filter_jit
def unrolled_loss_fn(model, m0, feats, targets, cond, demag, solver=None):
    """Mean one-step MSE along a K-step autoregressive rollout from m0, the
    gradient flowing through every step (each rematerialised on the backward
    pass, so memory stays at about one step's): unrolled training, which
    trains the model on the states its own errors lead to."""

    @jax.checkpoint
    def step(m, target):
        m = jax.vmap(model)(model_inputs(m, feats, demag, solver, cond), cond)
        return m, jnp.mean(jnp.square(m - target))

    _, losses = jax.lax.scan(step, m0, jnp.moveaxis(targets, 1, 0))
    return losses.mean()


@eqx.filter_jit
def update_unrolled_fn(model, batch, demag, optimizer, opt_state, solver=None):
    loss, grad = eqx.filter_value_and_grad(unrolled_loss_fn)(
        model, *batch, demag, solver
    )
    updates, opt_state = optimizer.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit(donate="all-except-first")
def update_fn(model: eqx.Module, batch: PyTree, optimizer, opt_state):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, *batch)
    updates, opt_state = optimizer.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
