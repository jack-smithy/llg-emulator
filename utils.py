import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from einops import rearrange, reduce
from jaxtyping import PyTree
from torch.utils.data import default_collate
from tqdm import tqdm

from model import ClosureEmulator

# `batch["constant_scalars"]` in the order the generator writes them
# (datagen.generate_varied_field.create_split); main.py checks the dataset agrees.
SCALARS = ("Ms", "A", "alpha", "Hx", "Hy", "Hz")
MU_0 = 4e-7 * np.pi


def with_coords(m, grid):
    """Append the cell-centre coordinates in um as two channels.

    m: (B, 3, Lx, Ly); grid: (B, Lx, Ly, 2), `space_grid` as WellDataset serves
    it, i.e. the file's own (x, y) cell centres in metres. Taking it from the
    sample puts every geometry on the training set's scale.
    returns: (B, 5, Lx, Ly)
    """
    coords = rearrange(grid, "B Lx Ly D -> B D Lx Ly") * 1e6
    return jnp.concatenate((m, coords), axis=1)


def numpy_collate(batch):
    """
    Collate function specifies how to combine a list of data samples into a batch.
    default_collate creates pytorch tensors, then tree_map converts them into numpy arrays.
    """
    return jtu.tree_map(np.asarray, default_collate(batch))


def conditioning(scalars, grid):
    """`constant_scalars` (B, 6) + `space_grid` (B, Lx, Ly, 2) -> `cond` (B, 3):
    `[Hx / Ms, Hy / Ms, log(delta / l_ex)]`.

    Hz is always zero in the data and Ms is constant, so the field components are
    the whole per-sample variation, nondimensionalised. `delta / l_ex` is the cell
    size over the exchange length `sqrt(2A / (mu0 Ms^2))` (~0.88 for the native
    5 nm cells), the one length ratio of the discretised problem; a log because
    it is a scale parameter.
    """
    h = scalars[:, 3:5] / scalars[:, :1]
    delta = grid[:, 1, 0, 0] - grid[:, 0, 0, 0]  # x-spacing; in-plane cells are square
    l_ex = jnp.sqrt(2 * scalars[:, 1] / (MU_0 * scalars[:, 0] ** 2))
    return jnp.concatenate([h, jnp.log(delta / l_ex)[:, None]], axis=1)


def downsample(m, grid, factor: int):
    """Coarse-grain m and grid onto the same film with `factor`-times larger cells.

    m: (B, 3, Lx, Ly) unit vectors; grid: (B, Lx, Ly, 2) cell centres in metres;
    Lx and Ly must be multiples of `factor`. Each coarse cell's m is the mean of
    its `factor` x `factor` block, renormalised back onto the unit sphere (lossy
    near domain walls, which is inherent to a larger cell); its centre is the
    block's mean centre.
    returns: (m, grid) on the coarse mesh.
    """
    if factor == 1:
        return m, grid
    m = reduce(m, "B F (x i) (y j) -> B F x y", "mean", i=factor, j=factor)
    grid = reduce(grid, "B (x i) (y j) D -> B x y D", "mean", i=factor, j=factor)
    return m / jnp.linalg.norm(m, axis=1, keepdims=True), grid


def d4(fields, scalars, g: int):
    """Element `g` (0..7) of the square film's symmetry group applied to a batch:
    rotate the grid by 90 degrees g % 4 times, then for g >= 4 mirror it in x.

    fields: (B, T, Lx, Ly, 3) frames of m as WellDataset serves them, Lx == Ly;
    scalars: (B, 6) `constant_scalars`, whose Hx, Hy, Hz turn with the film.
    m and H are axial vectors: a rotation turns their in-plane components with the
    grid; the mirror x -> -x keeps m_x and flips m_y, m_z. Exchange, demag and the
    LLG are invariant under all eight, so a transformed trajectory is another
    trajectory of the same dynamics -- for models without coordinate inputs, whose
    grid is left as it is.
    """
    k = g % 4
    fields = np.rot90(fields, k, axes=(2, 3))  # axis 2 towards 3: x -> y, +90 degrees
    c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[k]
    scalars = np.array(scalars, copy=True)
    x, y = fields[..., 0], fields[..., 1]
    fields = np.stack([c * x - s * y, s * x + c * y, fields[..., 2]], axis=-1)
    hx, hy = scalars[:, 3], scalars[:, 4]
    scalars[:, 3], scalars[:, 4] = c * hx - s * hy, s * hx + c * hy
    if g >= 4:
        fields = fields[:, :, ::-1] * np.array([1.0, -1.0, -1.0], fields.dtype)
        scalars[:, 4:6] *= -1
    return fields, scalars


def prepare_batch(batch, pool: int = 1, augment: int | None = None):
    """Batch -> `(ms, targets, cond, grid, h)` in the model's layout, coarse-grained by
    `pool` first and, with `augment`, transformed by that element of `d4`.

    ms: (B, n_in, 3, Lx, Ly), the input frames, oldest first (n_in = the model's
    `in_frames`); targets: (B, T, 3, Lx, Ly), the T frames after them (T = 1 for
    one-step training, n for an n-step unroll); cond: (B, 3); grid: (B, Lx, Ly, 2)
    cell centres; h: (B, 3) applied field in A/m, what a solver takes.
    """
    grid0 = batch["space_grid"]
    inputs, outputs, scalars = (
        batch["input_fields"],
        batch["output_fields"],
        batch["constant_scalars"],
    )
    if augment is not None:
        inputs, scalars = d4(inputs, scalars, augment)
        outputs, _ = d4(outputs, batch["constant_scalars"], augment)

    def frames(x):  # (B, T, Lx, Ly, F) -> (B, T, F, Lx', Ly') on the pooled mesh
        T = x.shape[1]
        x = rearrange(x, "B T Lx Ly F -> (B T) F Lx Ly")
        x, grid = downsample(x, np.repeat(grid0, T, axis=0), pool)
        return rearrange(x, "(B T) F Lx Ly -> B T F Lx Ly", T=T), grid[::T]

    ms, grid = frames(inputs)
    targets, _ = frames(outputs)
    return ms, targets, conditioning(scalars, grid), grid, scalars[:, 3:6]


def model_step(model, ms, cond, grid, h, physics=None):
    """One emulator step for a batch: frames ms (B, n_in, 3, Lx, Ly), oldest first ->
    the next frame (B, 3, Lx, Ly).

    `physics` is the `physics.MeshPhysics` of this mesh. With its `solver`, a
    `model.ClosureEmulator` steps that solver under its closure field; any other model
    is solver-in-the-loop (Um et al. 2020): the coarse solver steps the latest frame
    under the applied field h (B, 3) and the network corrects its output,
    m' = P(m) + C(P(m)) (`model.ResidualEmulator` adds C to its input's first 3
    channels). Otherwise the network sees the latest frame first, the older ones
    newest first, the cell coordinates and, with `physics.demag`, the latest frame's
    demag field.
    """
    m = ms[:, -1]
    if physics is not None and physics.solver is not None:
        if isinstance(model, ClosureEmulator):
            return jax.vmap(lambda m, c, h: model(m, c, h, physics.solver))(m, cond, h)
        return jax.vmap(model)(jax.vmap(physics.solver)(m, h), cond)
    older = rearrange(ms[:, -2::-1], "B T F Lx Ly -> B (T F) Lx Ly")
    x = with_coords(jnp.concatenate([m, older], axis=1), grid)
    if physics is not None and physics.demag is not None:
        x = jnp.concatenate([x, jax.vmap(physics.demag)(m)], axis=1)
    return jax.vmap(model)(x, cond)


def advance(ms, m):
    """Slide the frame window: drop the oldest of ms (B, n_in, ...), append m."""
    return jnp.concatenate([ms[:, 1:], m[:, None]], axis=1)


@eqx.filter_jit
def predict(model, ms, cond, grid, h, physics=None):
    """`model_step`, jitted."""
    return model_step(model, ms, cond, grid, h, physics)


@eqx.filter_jit
def rollout(model, x, cond, grid, n_steps: int, h=None, physics=None):
    """Autoregressively predict n_steps frames after the context frames x.

    x: (B, n_in, Lx, Ly, F), as WellDataset serves input_fields, oldest first.
    cond: (B, 3), as `conditioning` returns it.
    grid: (B, Lx, Ly, 2), as WellDataset serves space_grid.
    h, physics: the applied field (B, 3) in A/m and the mesh's `physics.MeshPhysics`
    (`model_step`).
    returns: (B, n_steps, Lx, Ly, F), same layout as output_fields.
    """
    ms0 = rearrange(x, "B T Lx Ly F -> B T F Lx Ly")

    def step(ms, _):
        m = model_step(model, ms, cond, grid, h, physics)
        return advance(ms, m), m

    _, frames = jax.lax.scan(step, ms0, None, length=n_steps)
    return rearrange(frames, "T B F Lx Ly -> B T Lx Ly F")


def one_step_preds(model, loader, physics=None):
    """Teacher-forced one-step predictions over a whole loader.

    returns: (pred, truth), both (N, 1, Lx, Ly, F) numpy -- the layout the
    well's metrics reduce over.
    """
    preds, truths = [], []
    for batch in tqdm(loader, mininterval=10):
        ms, targets, cond, grid, h = prepare_batch(batch)
        preds.append(np.asarray(predict(model, ms, cond, grid, h, physics)))
        truths.append(targets[:, 0])

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
def loss_fn(model, ms, targets, cond, grid, h, physics=None):
    """Mean squared error over an autoregressive unroll of `targets.shape[1]` steps
    after the context frames ms, each step fed the previous predictions (one step:
    plain one-step loss). With a solver, gradients flow back through it: the
    solver-in-the-loop loss."""

    if targets.shape[1] == 1:
        # one step: no scan, whose stacked residuals keep every intermediate of the
        # step alive (~16 GiB more at 32 x 256^2 than XLA's own fused schedule)
        m = model_step(model, ms, cond, grid, h, physics)
        return jnp.mean(jnp.square(m - targets[:, 0]))

    def step(ms, target):
        m = model_step(model, ms, cond, grid, h, physics)
        return advance(ms, m), jnp.mean(jnp.square(m - target))

    # an unroll is rematerialised per step: backprop keeps only each step's input
    # state instead of every step's solver and network activations (an 8-step unroll
    # at 128^2 needs ~43 GiB without), at the cost of recomputing each step's forward
    # pass
    step = jax.checkpoint(step)

    _, losses = jax.lax.scan(step, ms, jnp.moveaxis(targets, 1, 0))
    return jnp.mean(losses)


# no donation: the physics' demag tensors are reused across steps
@eqx.filter_jit
def update_fn(model: eqx.Module, batch: PyTree, optimizer, opt_state, physics=None):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, *batch, physics)
    updates, opt_state = optimizer.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
